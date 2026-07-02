# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents custom agent used by Terminal Two backends.

The public HEP surface is LangChain Deep Agents harness engineering. This
module is the backend bridge: it lets an external task runner provide a live
sandbox while Deep Agents supplies the harness, tools, profile, and model loop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from langchain_core.messages import BaseMessage, message_to_dict
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

from deepagents import create_deep_agent
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox


class TerminalTwoSandboxBackend(BaseSandbox):
    """Deep Agents sandbox backend backed by a Terminal Two task environment."""

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        cwd: str = "/app",
        default_timeout: int = 120,
        output_limit: int = 200_000,
    ) -> None:
        self._environment = environment
        self._cwd = cwd
        self._default_timeout = default_timeout
        self._output_limit = output_limit

    @property
    def id(self) -> str:
        return f"terminal-two:{getattr(self._environment, 'context_id', 'environment')}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return asyncio.run(self.aexecute(command, timeout=timeout))

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        timeout_sec = self._default_timeout if timeout is None else timeout
        result = await self._environment.exec(
            command,
            cwd=self._cwd,
            timeout_sec=timeout_sec if timeout_sec and timeout_sec > 0 else None,
        )
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        truncated = False
        if len(output) > self._output_limit:
            output = (
                output[: self._output_limit]
                + "\n\n[Output truncated by Deep Agents Terminal Two adapter.]"
            )
            truncated = True
        return ExecuteResponse(output=output, exit_code=result.return_code, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return asyncio.run(self.aupload_files(files))

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for remote_path, content in files:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                try:
                    await self._environment.upload_file(tmp_path, remote_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                responses.append(FileUploadResponse(path=remote_path))
            except Exception as exc:  # noqa: BLE001 - preserve per-file partial failures
                responses.append(FileUploadResponse(path=remote_path, error=str(exc)))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return asyncio.run(self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for remote_path in paths:
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    await self._environment.download_file(remote_path, tmp_path)
                    content = tmp_path.read_bytes()
                finally:
                    tmp_path.unlink(missing_ok=True)
                responses.append(FileDownloadResponse(path=remote_path, content=content))
            except Exception as exc:  # noqa: BLE001 - preserve per-file partial failures
                responses.append(FileDownloadResponse(path=remote_path, error=str(exc)))
        return responses


class DeepAgentsTerminalTwoAgent(BaseAgent):
    """Run LangChain Deep Agents as a Terminal Two custom agent."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        api_base: str | None = None,
        profile_file: str | None = None,
        profile_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        recursion_limit: int = 100,
        cwd: str = "/app",
        execute_timeout: int = 120,
        store_all_messages: bool = False,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, **kwargs)
        if not model_name:
            raise ValueError("model_name is required")
        self.api_base = api_base
        self.profile_file = profile_file
        self.profile_key = profile_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.recursion_limit = recursion_limit
        self.cwd = cwd
        self.execute_timeout = execute_timeout
        self.store_all_messages = store_all_messages
        self.system_prompt = system_prompt

    @staticmethod
    def name() -> str:
        return "deepagents"

    def version(self) -> str | None:
        try:
            from deepagents._version import __version__

            return __version__
        except Exception:  # noqa: BLE001 - best-effort metadata
            return None

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model_name = _strip_litellm_openai_prefix(self.model_name or "")
        profile_loaded = _register_profile_file(self.profile_file)

        model_kwargs: dict[str, Any] = {
            "model": model_name,
            "base_url": self.api_base,
            "api_key": os.environ.get("OPENAI_API_KEY") or os.environ.get("NVIDIA_API_KEY"),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        model = ChatOpenAI(**{key: value for key, value in model_kwargs.items() if value is not None})

        backend = TerminalTwoSandboxBackend(
            environment,
            cwd=self.cwd,
            default_timeout=self.execute_timeout,
        )
        prompt = self.system_prompt or _terminal_two_system_prompt(self.cwd)
        agent = create_deep_agent(
            model=model,
            backend=backend,
            system_prompt=prompt,
            debug=False,
        )

        run_status = "completed"
        run_error = None
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": instruction}]},
                config={"recursion_limit": self.recursion_limit},
            )
        except GraphRecursionError as exc:
            run_status = "graph_recursion_limit"
            run_error = str(exc)
            result = {"messages": []}

        messages = list(result.get("messages") or [])
        usage = _collect_usage(messages)
        context.n_input_tokens = usage["input"]
        context.n_output_tokens = usage["output"]
        context.n_cache_tokens = usage["cache"]
        context.cost_usd = None
        context.metadata = {
            "profile_file": self.profile_file,
            "profile_loaded": profile_loaded,
            "profile_key": self.profile_key,
            "model": model_name,
            "api_base": self.api_base,
            "recursion_limit": self.recursion_limit,
            "cwd": self.cwd,
            "message_count": len(messages),
            "final_message": _last_text_message(messages),
            "run_status": run_status,
        }
        if run_error:
            context.metadata["run_error"] = run_error
        if self.store_all_messages:
            context.metadata["all_messages"] = [_message_to_jsonable(message) for message in messages]


def _register_profile_file(profile_file: str | None) -> bool:
    if not profile_file:
        return False
    path = Path(profile_file).resolve()
    spec = importlib.util.spec_from_file_location("_hep_terminal_two_profile", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import profile file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    register = getattr(module, "register", None)
    if register is None:
        raise ValueError(f"profile file has no register() function: {path}")
    register()
    return True


def _strip_litellm_openai_prefix(model_name: str) -> str:
    return model_name.removeprefix("openai/")


def _terminal_two_system_prompt(cwd: str) -> str:
    return f"""You are running a Terminal Two task through the LangChain Deep Agents harness.

The task environment is a sandbox. The working directory for shell execution is `{cwd}`.
Use the built-in filesystem tools and the `execute` tool to inspect, modify, test, and verify the task.

Important execution details:
- Each `execute` call is a fresh shell command. Do not rely on `cd` persisting across calls.
- Prefer absolute paths under `{cwd}`, or include the directory in a single command.
- Complete the requested task in the filesystem; the verifier will inspect the sandbox after you finish.
- Before finalizing, run the relevant check or command that demonstrates the task is complete.
"""


def _collect_usage(messages: list[Any]) -> dict[str, int]:
    usage = {"input": 0, "output": 0, "cache": 0}
    for message in messages:
        metadata = getattr(message, "usage_metadata", None) or {}
        if isinstance(metadata, dict):
            usage["input"] += int(metadata.get("input_tokens") or 0)
            usage["output"] += int(metadata.get("output_tokens") or 0)
            input_details = metadata.get("input_token_details") or {}
            if isinstance(input_details, dict):
                usage["cache"] += int(input_details.get("cache_read") or 0)
                usage["cache"] += int(input_details.get("cache_creation") or 0)
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None
        if isinstance(token_usage, dict) and not metadata:
            usage["input"] += int(token_usage.get("prompt_tokens") or 0)
            usage["output"] += int(token_usage.get("completion_tokens") or 0)
            prompt_details = token_usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                usage["cache"] += int(prompt_details.get("cached_tokens") or 0)
    return usage


def _last_text_message(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content[-4000:]
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            text = "\n".join(part for part in parts if part)
            if text:
                return text[-4000:]
    return None


def _message_to_jsonable(message: BaseMessage) -> dict[str, Any]:
    try:
        return message_to_dict(message)
    except Exception:  # noqa: BLE001 - debug metadata should not fail the run
        return {
            "type": type(message).__name__,
            "content": getattr(message, "content", None),
        }
