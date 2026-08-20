#!/usr/bin/env bash
# Run the semvec-atlas advisor demo with the right virtualenv.
#
# Resolution order:
#   1. $SEMVEC_ATLAS_VENV (explicit override)
#   2. newest ~/.venvs/semvec-atlas* that has semvec_atlas installed
# Extra args are passed through (e.g. --no-llm, --keep-db).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

pick_venv() {
    if [[ -n "${SEMVEC_ATLAS_VENV:-}" ]]; then
        echo "$SEMVEC_ATLAS_VENV"; return
    fi
    for v in $(ls -d "$HOME"/.venvs/semvec-atlas* 2>/dev/null | sort -rV); do
        if "$v/bin/python" -c "import semvec_atlas, pymongo, sentence_transformers" 2>/dev/null; then
            echo "$v"; return
        fi
    done
    echo ""
}

venv="$(pick_venv)"
if [[ -z "$venv" ]]; then
    echo "No suitable virtualenv found." >&2
    echo "Create one with:" >&2
    echo "  python3 -m venv ~/.venvs/semvec-atlas" >&2
    echo "  ~/.venvs/semvec-atlas/bin/pip install -e '$here/..[langgraph,compliance,mcp,dev]' sentence-transformers openai tiktoken" >&2
    exit 1
fi

if ! docker ps --filter name=semvec-atlas-mongo --format '{{.Status}}' | grep -q Up; then
    echo "Starting MongoDB container semvec-atlas-mongo …"
    docker start semvec-atlas-mongo >/dev/null 2>&1 || docker run -d --name semvec-atlas-mongo -p 27017:27017 mongodb/mongodb-atlas-local:latest >/dev/null
fi

echo "venv: $venv"
exec "$venv/bin/python" "$here/advisor_demo.py" "$@"
