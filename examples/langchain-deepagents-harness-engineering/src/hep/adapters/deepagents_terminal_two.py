# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HarnessAdapter that evaluates LangChain Deep Agents in sandboxed Terminal Two tasks."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hep.adapters.base import HarnessAdapter

_DEFAULT_DATASET = "terminal-bench/terminal-bench-2-1"
_MAX_FAILURE_CHARS = 4000
_MAX_CONTEXT_CHARS = 8000

_CONFIG_SURFACE = """\
A Terminal Two HarnessProfile is a normal LangChain Deep Agents profile file. It must expose
register(), and register() should call deepagents.register_harness_profile(...) for the model
key under test, for example openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B.

Profile levers:
  - system_prompt_suffix: use sparingly for general planning, grounding, completion, and
    Terminal Two workflow guidance that should apply across tasks.
  - tool_description_overrides: use when a tool's default description omits behavior the model
    must know to use it correctly.
  - extra_middleware: use for provider compatibility shims, tool-result annotations, retry
    wrappers, and broad behavioral guardrails that are easier to enforce in code than prompt text.

Avoid fitting to exact task names, expected answers, or a single failing trace. The useful
profile changes should explain a broadly reusable failure mode and should be re-run against
previously passing tasks to check for regressions.
"""


def _warn(message: str) -> None:
    print(f"[hep:deepagents-terminal-two] warning: {message}", file=sys.stderr, flush=True)


def _log(message: str) -> None:
    print(f"[hep:deepagents-terminal-two] {message}", flush=True)


class DeepAgentsTerminalTwoAdapter(HarnessAdapter):
    """Run Deep Agents Terminal Two tasks and expose results to HEP optimizers.

    Args:
        dataset: Terminal Two dataset identifier accepted by the runner.
        jobs_dir: Directory where Terminal Two jobs are written.
        api_base: OpenAI-compatible endpoint base URL for the evaluated model.
        n_attempts: Number of attempts per task.
        n_concurrent: Number of concurrent tasks.
        recursion_limit: LangGraph recursion limit for the Deep Agents run.
        execute_timeout: Timeout, in seconds, for sandbox shell commands.
        max_retries: Terminal Two runner retries for infrastructure failures.
        runner_bin: Terminal Two runner executable. Defaults to ``harbor``.
    """

    def __init__(
        self,
        *,
        dataset: str = _DEFAULT_DATASET,
        jobs_dir: Path = Path("jobs"),
        api_base: str | None = None,
        n_attempts: int = 1,
        n_concurrent: int = 1,
        recursion_limit: int = 100,
        execute_timeout: int = 120,
        max_retries: int = 1,
        runner_bin: str = "harbor",
    ) -> None:
        self.dataset = dataset
        self.jobs_dir = jobs_dir.resolve()
        self.api_base = api_base
        self.n_attempts = n_attempts
        self.n_concurrent = n_concurrent
        self.recursion_limit = recursion_limit
        self.execute_timeout = execute_timeout
        self.max_retries = max_retries
        self.runner_bin = runner_bin
        self._root = Path(__file__).resolve().parents[3]

    def read_config(self, config_file: Path) -> str:
        return config_file.read_text(encoding="utf-8")

    def write_config(self, config_file: Path, content: str) -> None:
        config_file.write_text(content, encoding="utf-8")

    def validate_config(self, content: str) -> bool:
        try:
            tree = ast.parse(content)
        except SyntaxError as exc:
            _warn(f"proposed profile has syntax error: {exc}")
            return False
        if not any(isinstance(node, ast.FunctionDef) and node.name == "register" for node in tree.body):
            _warn("proposed profile does not define register()")
            return False
        return True

    def run_eval_suite(
        self,
        model: str,
        config_file: Path,
        report_file: Path,
        categories: list[str] | None,
    ) -> dict[str, Any]:
        """Run a task subset; ``categories`` are treated as Terminal Two task IDs."""

        tasks = categories or []
        job_name = self._job_name(model=model, config_file=config_file, tasks=tasks, suffix="suite")
        _log(f"running Terminal Two suite job={job_name} tasks={tasks or 'dataset default'}")
        job_dir = self._run_job(
            model=model,
            config_file=config_file,
            tasks=tasks,
            job_name=job_name,
        )
        report = self._parse_job(job_dir)
        self._write_report(report_file, report)
        return report

    def run_single_test(
        self,
        node_id: str,
        model: str,
        config_file: Path,
        report_file: Path,
    ) -> tuple[bool, str]:
        job_name = self._job_name(model=model, config_file=config_file, tasks=[node_id], suffix="single")
        _log(f"re-running Terminal Two task job={job_name} task={node_id}")
        job_dir = self._run_job(
            model=model,
            config_file=config_file,
            tasks=[node_id],
            job_name=job_name,
        )
        report = self._parse_job(job_dir)
        self._write_report(report_file, report)
        failures = report.get("failures") or []
        return report.get("failed", 0) == 0, failures[0].get("failure_message", "") if failures else ""

    def extract_task_source(self, task_name: str) -> str | None:
        return (
            "Terminal Two tasks are dataset-defined. Use the task id, job artifact, "
            "failure trace, and sandbox outputs as the source of task context.\n"
            f"Task id: {task_name}"
        )

    def collect_context(self, config_file: Path, failure_message: str) -> str:
        sections = [
            "# Current profile file\n" + _truncate(config_file.read_text(encoding="utf-8"), _MAX_CONTEXT_CHARS),
            "# Terminal Two adapter surface\n" + _CONFIG_SURFACE,
        ]
        agent_file = self._root / "src" / "hep" / "deepagents_terminal_two_agent.py"
        if agent_file.is_file():
            sections.append(
                "# Deep Agents Terminal Two agent\n"
                + _truncate(agent_file.read_text(encoding="utf-8"), _MAX_CONTEXT_CHARS)
            )
        return "\n\n---\n\n".join(sections)

    def describe_config_surface(self) -> str:
        return _CONFIG_SURFACE

    def _run_job(
        self,
        *,
        model: str,
        config_file: Path,
        tasks: list[str],
        job_name: str,
    ) -> Path:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.runner_bin,
            "run",
            "-d",
            self.dataset,
            "-a",
            "hep.deepagents_terminal_two_agent:DeepAgentsTerminalTwoAgent",
            "-m",
            model,
            "-n",
            str(self.n_concurrent),
            "-k",
            str(self.n_attempts),
            "--job-name",
            job_name,
            "--yes",
            "--max-retries",
            str(self.max_retries),
            "--ak",
            f"profile_file={config_file.resolve()}",
            "--ak",
            f"recursion_limit={self.recursion_limit}",
            "--ak",
            f"execute_timeout={self.execute_timeout}",
            "--ak",
            "store_all_messages=false",
        ]
        if self.api_base:
            cmd += ["--ak", f"api_base={self.api_base}"]
        for task in tasks:
            cmd += ["-i", task]

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{self._root / 'src'}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(self._root / "src")
        )
        result = subprocess.run(cmd, cwd=self._root, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Terminal Two runner failed with exit code {result.returncode}: {' '.join(cmd)}")

        job_dir = self.jobs_dir / job_name
        if not (job_dir / "result.json").is_file():
            raise RuntimeError(f"Terminal Two runner did not write {job_dir / 'result.json'}")
        return job_dir

    def _parse_job(self, job_dir: Path) -> dict[str, Any]:
        result_path = job_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        stats = result.get("stats") or {}
        evals = stats.get("evals") or {}

        failures: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        passed = 0
        failed = 0
        for trial_result in sorted(job_dir.glob("*/result.json")):
            data = json.loads(trial_result.read_text(encoding="utf-8"))
            task_name = data.get("task_name") or data.get("trial_name") or trial_result.parent.name
            reward = _extract_reward(data)
            agent_result = data.get("agent_result") or {}
            verifier_result = data.get("verifier_result") or {}
            metadata = agent_result.get("metadata") or {}
            row = {
                "trial": data.get("trial_name", trial_result.parent.name),
                "task_name": task_name,
                "reward": reward,
                "input_tokens": agent_result.get("n_input_tokens"),
                "cache_tokens": agent_result.get("n_cache_tokens"),
                "output_tokens": agent_result.get("n_output_tokens"),
                "run_status": metadata.get("run_status"),
            }
            trial_rows.append(row)
            if reward == 1.0:
                passed += 1
            else:
                failed += 1
                failures.append(
                    {
                        "test_name": task_name,
                        "failure_message": _failure_message(
                            data=data,
                            reward=reward,
                            agent_result=agent_result,
                            verifier_result=verifier_result,
                        ),
                        "trial": row["trial"],
                        "reward": reward,
                    }
                )

        if not trial_rows and evals:
            for task_name, task_data in evals.items():
                metrics = task_data.get("metrics") or [{}]
                mean = metrics[0].get("mean")
                if mean == 1.0:
                    passed += int(task_data.get("n_trials") or 1)
                else:
                    failed += int(task_data.get("n_trials") or 1)
                    failures.append(
                        {
                            "test_name": task_name,
                            "failure_message": f"Terminal Two mean reward={mean}",
                            "reward": mean,
                        }
                    )

        return {
            "passed": passed,
            "failed": failed,
            "failures": failures,
            "job_dir": str(job_dir),
            "summary": {
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "completed_trials": stats.get("n_completed_trials"),
                "errored_trials": stats.get("n_errored_trials"),
                "retries": stats.get("n_retries"),
                "input_tokens": stats.get("n_input_tokens"),
                "cache_tokens": stats.get("n_cache_tokens"),
                "output_tokens": stats.get("n_output_tokens"),
                "cost_usd": stats.get("cost_usd"),
            },
            "trials": trial_rows,
        }

    @staticmethod
    def _write_report(report_file: Path, report: dict[str, Any]) -> None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _job_name(*, model: str, config_file: Path, tasks: list[str], suffix: str) -> str:
        model_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", model.removeprefix("openai/")).strip("-")
        profile_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", config_file.stem).strip("-")
        task_slug = f"{len(tasks)}tasks" if tasks else "all"
        return f"hep-deepagents-terminal-two-{profile_slug}-{model_slug}-{task_slug}-{suffix}-{int(time.time())}"


def _extract_reward(data: dict[str, Any]) -> Any:
    verifier = data.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if "reward" in rewards:
        return rewards["reward"]
    agent_result = data.get("agent_result") or {}
    return agent_result.get("reward")


def _failure_message(
    *,
    data: dict[str, Any],
    reward: Any,
    agent_result: dict[str, Any],
    verifier_result: dict[str, Any],
) -> str:
    pieces = [
        f"Task: {data.get('task_name') or data.get('trial_name')}",
        f"Reward: {reward}",
    ]
    metadata = agent_result.get("metadata") or {}
    for key in ("run_status", "run_error", "final_message"):
        if metadata.get(key):
            pieces.append(f"{key}: {metadata[key]}")
    if verifier_result:
        pieces.append("verifier_result: " + _truncate(json.dumps(verifier_result, indent=2), 2000))
    return _truncate("\n\n".join(pieces), _MAX_FAILURE_CHARS)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... [truncated]"
