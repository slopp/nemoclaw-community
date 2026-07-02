#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v harbor >/dev/null 2>&1; then
  uv tool install harbor
fi

HARBOR_PY="${HARBOR_PY:-$HOME/.local/share/uv/tools/harbor/bin/python}"
if [[ ! -x "$HARBOR_PY" ]]; then
  echo "could not find Harbor tool Python at $HARBOR_PY" >&2
  echo "set HARBOR_PY to the Python executable used by your harbor tool install" >&2
  exit 1
fi

"$HARBOR_PY" -m pip install --upgrade deepagents langchain-openai
harbor --help >/dev/null

echo "Deep Agents Terminal Two runner dependencies are installed in: $HARBOR_PY"
