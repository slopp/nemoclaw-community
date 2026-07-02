# LangChain Deep Agents Terminal Two Adapter

This fork keeps HEP generic while showing how to add a second eval suite for the same target
framework. The stock HEP path runs LangChain Deep Agents pytest evals. This adapter runs Terminal
Two tasks with the same Deep Agents `HarnessProfile` files.

The Terminal Two runner supplies task sandboxes and scoring; Deep Agents supplies the agent loop,
filesystem tools, model wiring, and profile under test. The Harbor custom-agent bridge is included
in this example so the adapter is self-contained.

## Shape

```
hep langchain-term2 run
  -> DeepAgentsTerminalTwoAdapter
     -> Terminal Two runner custom agent:
        hep.deepagents_terminal_two_agent:DeepAgentsTerminalTwoAgent
        -> deepagents.create_deep_agent(...)
```

The same `HarnessAdapter` contract used by `hep langchain ralph` is preserved:

- `run_eval_suite()` runs a task list and writes a JSON report with `passed`, `failed`, and
  `failures`.
- `run_single_test()` re-runs one task for fix verification.
- `read_config()`, `write_config()`, and `validate_config()` operate on normal Deep Agents
  `HarnessProfile` Python files.

## Example

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

The command writes a Ralph-compatible JSON report and leaves the raw Terminal Two job under `jobs/`.

## Optimizer Use

The adapter is intentionally shaped so Ralph can use it without changing the optimizer loop.
The CLI exposes this as `hep langchain-term2 ralph`, and the equivalent Python wiring is:

```python
from hep.adapters.deepagents_terminal_two import DeepAgentsTerminalTwoAdapter
from hep.optimizers.ralph import RalphLoopOptimizer

adapter = DeepAgentsTerminalTwoAdapter(
    dataset="terminal-bench/terminal-bench-2-1",
    api_base="https://nemotron-ultra.apps.inference-tme.nvidia.com/v1",
    n_attempts=1,
    n_concurrent=1,
)
optimizer = RalphLoopOptimizer(ralph_model="anthropic:claude-opus-4-8")
optimizer.run(
    adapter=adapter,
    model="openai:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    config_file=Path("examples/deepagents/profiles/nvidia_nemotron_3_ultra_base.py"),
    categories=["terminal-bench/regex-log"],
    max_iters=3,
    verify_runs=1,
)
```

## Harbor Bridge

`src/hep/deepagents_terminal_two_agent.py` contains the Harbor custom-agent implementation:

- `TerminalTwoSandboxBackend` adapts Harbor's environment API to Deep Agents' sandbox backend
  protocol.
- `DeepAgentsTerminalTwoAgent` creates the LangChain model, imports the profile file, calls
  `deepagents.create_deep_agent(...)`, and records token usage plus run metadata back into the job.

The HEP adapter treats Harbor as the runner implementation. The public extension point remains
`HarnessAdapter`: translate a suite's native job artifacts into `passed`, `failed`, and `failures`.
