# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`hep langchain` — commands that target the LangChain deepagents SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from hep.adapters.deepagents import DeepAgentsAdapter
from hep.adapters.deepagents_terminal_two import DeepAgentsTerminalTwoAdapter
from hep.optimizers.ralph import _DEFAULT_RALPH_MAX_TURNS

_EXAMPLE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EVALS_DIR = (
    _EXAMPLE_ROOT.parent / "harness-engineering-playground" / "external" / "deepagents" / "libs" / "evals"
)
"""Shared deepagents submodule's libs/evals from the original HEP example."""

app = typer.Typer(name="langchain", help="Commands targeting the LangChain deepagents SDK.")


@app.command("ralph")
def ralph(
    model: Annotated[str, typer.Option(help="Model spec under evaluation, e.g. openai:nvidia/...")],
    profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Path to the HarnessProfile Python file to improve.",
        ),
    ],
    evals_dir: Annotated[
        Path,
        typer.Option(
            envvar="HEP_EVALS_DIR",
            exists=True,
            file_okay=False,
            help="Path to a deepagents checkout's libs/evals dir.",
        ),
    ] = _DEFAULT_EVALS_DIR,
    ralph_model: Annotated[
        str,
        typer.Option(
            "--ralph-model",
            envvar="RALPH_MODEL",
            help="Provider-prefixed model for the agentic proposer, e.g. anthropic:claude-opus-4-8.",
        ),
    ] = "anthropic:claude-opus-4-8",
    ralph_max_turns: Annotated[
        int,
        typer.Option(envvar="RALPH_MAX_TURNS", help="Max tool-call turns per proposer session."),
    ] = _DEFAULT_RALPH_MAX_TURNS,
    ralph_base_url: Annotated[
        str | None,
        typer.Option(
            envvar="RALPH_BASE_URL",
            help="Custom Anthropic-compatible base URL for the agentic proposer.",
        ),
    ] = None,
    category: Annotated[
        list[str], typer.Option("--category", help="Restrict evals to this category (repeatable).")
    ] = [],  # noqa: B006 - typer requires a literal default for multi-value options
    max_iters: Annotated[int, typer.Option(help="Max improvement iterations across all failures.")] = 5,
    max_iters_per_failure: Annotated[
        int, typer.Option(help="Max fix attempts per failing test per round.")
    ] = 3,
    verify_runs: Annotated[
        int, typer.Option(help="Consecutive passing runs required to confirm a fix.")
    ] = 1,
) -> None:
    """Run the ralph loop with a tool-using agentic proposer instead of a single-shot completion."""
    try:
        from hep.optimizers._agentic_proposer import ensure_deepagents_available

        ensure_deepagents_available()
        from hep.optimizers.ralph import RalphLoopOptimizer
    except ImportError as exc:
        typer.echo(
            "error: `hep langchain ralph` requires the `deepagents` extra "
            "(deepagents, langchain, langchain-anthropic). Install it with:\n"
            "  uv sync --extra deepagents\n"
            f"Import error: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    adapter = DeepAgentsAdapter(evals_dir=evals_dir)
    optimizer = RalphLoopOptimizer(
        ralph_model=ralph_model,
        ralph_max_turns=ralph_max_turns,
        ralph_base_url=ralph_base_url,
    )
    result = optimizer.run(
        adapter=adapter,
        model=model,
        config_file=profile.resolve(),
        categories=category or None,
        max_iters=max_iters,
        max_iters_per_failure=max_iters_per_failure,
        verify_runs=verify_runs,
    )
    raise typer.Exit(code=0 if result.succeeded or result.baseline_failed == 0 else 1)


@app.command("terminal-two")
def terminal_two(
    model: Annotated[str, typer.Option(help="Model spec under evaluation, e.g. openai:nvidia/...")],
    profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Path to the HarnessProfile Python file to apply.",
        ),
    ],
    task: Annotated[
        list[str],
        typer.Option(
            "--task",
            help="Terminal Two task id to run (repeatable). Omit to use the dataset default selection.",
        ),
    ] = [],  # noqa: B006 - typer requires a literal default for multi-value options
    dataset: Annotated[
        str,
        typer.Option(envvar="HEP_TERMINAL_TWO_DATASET", help="Terminal Two dataset id."),
    ] = "terminal-bench/terminal-bench-2-1",
    api_base: Annotated[
        str | None,
        typer.Option(envvar="OPENAI_BASE_URL", help="OpenAI-compatible endpoint base URL."),
    ] = None,
    report: Annotated[
        Path,
        typer.Option(help="JSON report path to write."),
    ] = Path("reports/deepagents-terminal-two-report.json"),
    jobs_dir: Annotated[
        Path,
        typer.Option(help="Terminal Two jobs directory."),
    ] = Path("jobs"),
    n_attempts: Annotated[int, typer.Option("-k", help="Attempts per task.")] = 1,
    n_concurrent: Annotated[int, typer.Option("-n", help="Concurrent tasks.")] = 1,
    recursion_limit: Annotated[int, typer.Option(help="LangGraph recursion limit.")] = 100,
    execute_timeout: Annotated[int, typer.Option(help="Sandbox shell command timeout in seconds.")] = 120,
    max_retries: Annotated[int, typer.Option(help="Runner retries for infrastructure failures.")] = 1,
    runner_bin: Annotated[
        str,
        typer.Option(help="Terminal Two runner executable; Harbor is the current backend."),
    ] = "harbor",
) -> None:
    """Run LangChain Deep Agents with a profile against the Terminal Two suite."""

    adapter = _terminal_two_adapter(
        dataset=dataset,
        jobs_dir=jobs_dir,
        api_base=api_base,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        recursion_limit=recursion_limit,
        execute_timeout=execute_timeout,
        max_retries=max_retries,
        runner_bin=runner_bin,
    )
    result = adapter.run_eval_suite(
        model=model,
        config_file=profile.resolve(),
        report_file=report,
        categories=task or None,
    )
    typer.echo(json_summary(result))
    raise typer.Exit(code=0 if result.get("failed", 0) == 0 else 1)


@app.command("ralph-terminal-two")
def ralph_terminal_two(
    model: Annotated[str, typer.Option(help="Model spec under evaluation, e.g. openai:nvidia/...")],
    profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Path to the HarnessProfile Python file to improve.",
        ),
    ],
    task: Annotated[
        list[str],
        typer.Option(
            "--task",
            help="Terminal Two task id to run (repeatable). Omit to use the dataset default selection.",
        ),
    ] = [],  # noqa: B006 - typer requires a literal default for multi-value options
    dataset: Annotated[
        str,
        typer.Option(envvar="HEP_TERMINAL_TWO_DATASET", help="Terminal Two dataset id."),
    ] = "terminal-bench/terminal-bench-2-1",
    api_base: Annotated[
        str | None,
        typer.Option(envvar="OPENAI_BASE_URL", help="OpenAI-compatible endpoint base URL."),
    ] = None,
    jobs_dir: Annotated[
        Path,
        typer.Option(help="Terminal Two jobs directory."),
    ] = Path("jobs"),
    n_attempts: Annotated[int, typer.Option("-k", help="Attempts per task.")] = 1,
    n_concurrent: Annotated[int, typer.Option("-n", help="Concurrent tasks.")] = 1,
    recursion_limit: Annotated[int, typer.Option(help="LangGraph recursion limit.")] = 100,
    execute_timeout: Annotated[int, typer.Option(help="Sandbox shell command timeout in seconds.")] = 120,
    max_retries: Annotated[int, typer.Option(help="Runner retries for infrastructure failures.")] = 1,
    runner_bin: Annotated[
        str,
        typer.Option(help="Terminal Two runner executable; Harbor is the current backend."),
    ] = "harbor",
    ralph_model: Annotated[
        str,
        typer.Option(
            "--ralph-model",
            envvar="RALPH_MODEL",
            help="Provider-prefixed model for the agentic proposer, e.g. anthropic:claude-opus-4-8.",
        ),
    ] = "anthropic:claude-opus-4-8",
    ralph_max_turns: Annotated[
        int,
        typer.Option(envvar="RALPH_MAX_TURNS", help="Max tool-call turns per proposer session."),
    ] = _DEFAULT_RALPH_MAX_TURNS,
    ralph_base_url: Annotated[
        str | None,
        typer.Option(
            envvar="RALPH_BASE_URL",
            help="Custom Anthropic-compatible base URL for the agentic proposer.",
        ),
    ] = None,
    max_iters: Annotated[int, typer.Option(help="Max improvement iterations across all failures.")] = 5,
    max_iters_per_failure: Annotated[
        int, typer.Option(help="Max fix attempts per failing test per round.")
    ] = 3,
    verify_runs: Annotated[
        int, typer.Option(help="Consecutive passing runs required to confirm a fix.")
    ] = 1,
) -> None:
    """Run the Ralph optimization loop against the Terminal Two suite."""

    try:
        from hep.optimizers._agentic_proposer import ensure_deepagents_available

        ensure_deepagents_available()
        from hep.optimizers.ralph import RalphLoopOptimizer
    except ImportError as exc:
        typer.echo(
            "error: `hep langchain ralph-terminal-two` requires the `deepagents` extra "
            "(deepagents, langchain, langchain-anthropic). Install it with:\n"
            "  uv sync --extra deepagents --extra terminal-two\n"
            f"Import error: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    adapter = _terminal_two_adapter(
        dataset=dataset,
        jobs_dir=jobs_dir,
        api_base=api_base,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        recursion_limit=recursion_limit,
        execute_timeout=execute_timeout,
        max_retries=max_retries,
        runner_bin=runner_bin,
    )
    optimizer = RalphLoopOptimizer(
        ralph_model=ralph_model,
        ralph_max_turns=ralph_max_turns,
        ralph_base_url=ralph_base_url,
    )
    result = optimizer.run(
        adapter=adapter,
        model=model,
        config_file=profile.resolve(),
        categories=task or None,
        max_iters=max_iters,
        max_iters_per_failure=max_iters_per_failure,
        verify_runs=verify_runs,
    )
    raise typer.Exit(code=0 if result.succeeded or result.baseline_failed == 0 else 1)


def _terminal_two_adapter(
    *,
    dataset: str,
    jobs_dir: Path,
    api_base: str | None,
    n_attempts: int,
    n_concurrent: int,
    recursion_limit: int,
    execute_timeout: int,
    max_retries: int,
    runner_bin: str,
) -> DeepAgentsTerminalTwoAdapter:
    return DeepAgentsTerminalTwoAdapter(
        dataset=dataset,
        jobs_dir=jobs_dir,
        api_base=api_base,
        n_attempts=n_attempts,
        n_concurrent=n_concurrent,
        recursion_limit=recursion_limit,
        execute_timeout=execute_timeout,
        max_retries=max_retries,
        runner_bin=runner_bin,
    )


def json_summary(result: dict[str, object]) -> str:
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    return (
        f"passed={result.get('passed', 0)} failed={result.get('failed', 0)} "
        f"job_dir={result.get('job_dir', 'n/a')} "
        f"input_tokens={summary.get('input_tokens', 'n/a')} "
        f"cache_tokens={summary.get('cache_tokens', 'n/a')} "
        f"output_tokens={summary.get('output_tokens', 'n/a')}"
    )
