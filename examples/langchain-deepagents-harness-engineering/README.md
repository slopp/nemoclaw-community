# Harness Engineering Playground (`hep`) for LangChain Deep Agents

A CLI for automated, eval-driven harness profile improvement. This fork is intentionally generic:
it keeps LangChain Deep Agents as the target framework while showing two eval-suite adapters:
the stock Deep Agents pytest eval suite and a Terminal Two sandboxed eval suite.

`hep` is built around two extension seams:

- **`HarnessAdapter`** — knows how to run one target framework's eval suite and read/write/validate
  its config file. `DeepAgentsAdapter` (targeting the [LangChain deepagents SDK](https://github.com/langchain-ai/deepagents))
  targets the Deep Agents pytest eval suite, and `DeepAgentsTerminalTwoAdapter` targets sandboxed
  Terminal Two tasks through the same profile interface.
- **`Optimizer`** — a technique for searching over config edits to fix failing evals.
  - **`RalphLoopOptimizer`** (`hep langchain ralph`, `hep langchain-term2 ralph`) proposes a fix via a bounded
    tool-using deep-agent session built on LangChain's `deepagents` — the model gets real
    `read_file`/`edit_file`/`grep`/`glob` tools scoped to an isolated workspace, plus read-only
    access to the real SDK and eval-suite source trees, so it can investigate instead of guessing
    from a fixed snippet. It shares its propose/apply/verify/roll-back outer loop
    (`IterativeFixOptimizer`) with any other `Optimizer` implementation. Requires the
    `deepagents` extra (`uv sync --extra deepagents`).

An `Optimizer` only calls `HarnessAdapter` methods, never a specific framework's APIs directly —
so the same technique runs unmodified against any framework with an adapter, and a new technique
is immediately usable against every existing adapter.

This is the extension pattern the fork is meant to show: add an adapter for a second eval suite,
keep the same Deep Agents profile file contract, and reuse the optimizer loop.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An Anthropic API key (or an Anthropic-compatible endpoint) for the frontier model that proposes
  fixes
- The shared `deepagents` git submodule from the original HEP example
  (`../harness-engineering-playground/external/deepagents`), initialized via
  `git submodule update --init examples/harness-engineering-playground/external/deepagents`.
  It supplies `libs/evals`, the eval test files used by `hep langchain ralph` that aren't
  published in the `deepagents-evals` pip package. Set `--evals-dir`/`HEP_EVALS_DIR` to use a
  different checkout.
- The `deepagents` extra (`uv sync --extra deepagents`) — a hard requirement, not optional. It
  installs `deepagents`, `langchain`, `langchain-anthropic`, and `langchain-openai` into hep's own venv.
- The Terminal Two option also needs the runner-side dependencies installed into the Harbor tool
  environment. Run `./scripts/ensure_deepagents_terminal_two_deps.sh` after `uv sync --extra terminal-two`.

## Quickstart: Deep Agents Evals

```bash
cd examples/langchain-deepagents-harness-engineering
git -C ../.. submodule update --init examples/harness-engineering-playground/external/deepagents
uv sync --extra deepagents
cp .env.example .env   # fill in RALPH_API_KEY (or ANTHROPIC_API_KEY)
uv run hep langchain ralph \
  --model "openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B" \
  --profile examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py \
  --ralph-model anthropic:claude-opus-4-8 \
  --category file_operations \
  --max-iters 5 \
  --verify-runs 3
```

`--evals-dir` is omitted above — it defaults to the shared submodule. Pass
`--evals-dir /path/to/other-checkout/libs/evals` to point at a different deepagents checkout
instead (e.g. one with local, uncommitted profile tuning of your own).

See [docs/verify-functionality.md](docs/verify-functionality.md) to confirm your setup works
before running a real improvement loop, and the `Extending` section below for how to add a
new `Optimizer` or a new `HarnessAdapter`.

## Try it: optimize a blank profile from scratch

`examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py` is a deliberately blank
`HarnessProfile` (no prompt tuning, no middleware). Point `ralph` at it directly — no
copying, no editing the submodule checkout at all:

```bash
uv run hep langchain ralph \
  --model "openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B" \
  --profile examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py \
  --category file_operations \
  --max-iters 5
```

Watch it iteratively rediscover the kind of tuning already checked into deepagents' own built-in
`_nvidia_nemotron_3_ultra.py`.

## Quickstart: Terminal Two Eval Suite

Use `hep langchain-term2 run` when you want the same Deep Agents profile surface but a Terminal
Two task runner and sandbox instead of the Deep Agents pytest eval suite:

```bash
cd examples/langchain-deepagents-harness-engineering
uv sync --extra terminal-two
./scripts/ensure_deepagents_terminal_two_deps.sh

export OPENAI_API_KEY="$NVIDIA_API_KEY"
export OPENAI_BASE_URL="https://nemotron-ultra.apps.inference-tme.nvidia.com/v1"

uv run hep langchain-term2 run \
  --model "openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B" \
  --profile examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py \
  --task terminal-bench/regex-log \
  --task terminal-bench/filter-js-from-html \
  --report reports/deepagents-ultra-smoke.json \
  -n 1 \
  -k 1
```

The setup script installs the Terminal Two runner as a `uv` tool if needed and installs `deepagents`
plus `langchain-openai` into the runner's Python environment, which is where the custom agent is
imported. The command writes a Ralph-compatible JSON report with `passed`, `failed`, `failures`,
token summary fields, and the raw Terminal Two `job_dir`. See
[docs/langchain-deepagents-terminal-two-adapter.md](docs/langchain-deepagents-terminal-two-adapter.md)
for the adapter shape and how to wire it into `RalphLoopOptimizer`.

The CLI shape is `hep <target-framework/eval-suite> <action-or-optimizer>`. That keeps `ralph`
as the optimizer command for both eval suites:

```bash
uv run hep langchain ralph       # Deep Agents pytest eval suite
uv run hep langchain-term2 ralph # Terminal Two eval suite
```

Run the Terminal Two optimizer path directly:

```bash
uv run hep langchain-term2 ralph \
  --model "openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B" \
  --profile examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py \
  --task terminal-bench/regex-log \
  --max-iters 3 \
  --verify-runs 1
```

**This also works for a genuinely custom profile you write yourself** — the profile file doesn't
need to live inside the deepagents checkout, or be one of the SDK's built-in files. For pytest eval
runs, `--profile` is loaded by `hep_profile_plugin.py` before test collection. For Terminal Two
runs, `DeepAgentsTerminalTwoAgent` imports the same profile file before calling
`create_deep_agent(...)`. In both cases the profile uses the normal `register()` convention from
Deep Agents' built-in profile files, and no deepagents source edits are required.

## Layout

```
plugins/
└── hep_profile_plugin.py   — standalone pytest plugin; registers --profile's file before collection
examples/deepagents/profiles/
└── nvidia_nemotron_3_ultra_base.py   — reference blank-baseline profile, runnable as-is
../harness-engineering-playground/external/deepagents/
                            — shared langchain-ai/deepagents submodule; default --evals-dir target
src/hep/
├── main.py                 — root Typer app
├── adapters/
│   ├── base.py             — HarnessAdapter ABC
│   ├── deepagents.py       — Deep Agents pytest eval adapter
│   └── deepagents_terminal_two.py — Deep Agents Terminal Two eval-suite adapter
├── deepagents_terminal_two_agent.py — Harbor custom agent that calls create_deep_agent()
├── optimizers/
│   ├── base.py                  — Optimizer ABC + OptimizationResult + IterativeFixOptimizer
│   │                               (shared propose/apply/verify/roll-back outer loop)
│   ├── _proposer_workspace.py    — shared isolated-workspace scaffolding for agentic proposers
│   │                               (task.md/current/proposal.md/history layout)
│   ├── ralph.py                   — RalphLoopOptimizer (deepagents-based agentic proposer)
│   └── _agentic_proposer.py      — the one module that imports deepagents/LangChain; narrow on
│                                    purpose so the hard dependency's absence fails fast and
│                                    cleanly at the CLI boundary
└── cli/
    ├── langchain.py        — Deep Agents pytest eval-suite commands
    └── langchain_term2.py  — Deep Agents Terminal Two eval-suite commands
tests/unit_tests/           — adapter/optimizer tests against a FakeAdapter
```

## Extending

**New optimization technique for an existing framework** — implement `Optimizer`, add a CLI command
under the relevant `cli/<framework>.py` group. It can reuse any existing adapter.

**New eval suite for the same framework** — implement another `HarnessAdapter`, translate that
suite's native job artifacts into HEP's `passed`/`failed`/`failures` report shape, then register a
Typer group named for the framework+suite pair. Commands underneath that group are the actions or
optimizers, e.g. `hep langchain ralph` and `hep langchain-term2 ralph`.

**New target framework** — implement `HarnessAdapter`, add a `cli/<framework>.py` Typer group
registering it against `RalphLoopOptimizer` or any other existing `Optimizer`.
