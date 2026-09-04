#!/usr/bin/env bash
# Run the semvec-atlas advisor demo with the right virtualenv.
#
# Resolution order:
#   1. $SEMVEC_ATLAS_VENV (explicit override)
#   2. newest ~/.venvs/semvec-atlas* that has semvec_atlas installed
# Usage:
#   demo/run.sh                 six-act advisor demo (needs an LLM in .env)
#   demo/run.sh --no-llm        the same, without LLM calls
#   demo/run.sh --claims        claims-platform scenario (no LLM, seconds)
# Extra args are passed through (e.g. --keep-db).
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
    echo "  ~/.venvs/semvec-atlas/bin/pip install \\" >&2
    echo "      torch --index-url https://download.pytorch.org/whl/cpu" >&2
    echo "  ~/.venvs/semvec-atlas/bin/pip install \\" >&2
    echo "      -e '$here/..[langgraph,compliance,mcp,dev]' \\" >&2
    echo "      sentence-transformers openai tiktoken" >&2
    echo >&2
    echo "Install CPU torch first: the default wheels pull CUDA packages" >&2
    echo "that are large and unnecessary here, and can fail to unpack." >&2
    exit 1
fi

if ! docker ps --filter name=semvec-atlas-mongo --format '{{.Status}}' | grep -q Up; then
    echo "Starting MongoDB container semvec-atlas-mongo …"
    docker start semvec-atlas-mongo >/dev/null 2>&1 || docker run -d --name semvec-atlas-mongo -p 27017:27017 mongodb/mongodb-atlas-local:latest >/dev/null
fi

# Which demo? --claims runs the short claims-platform scenario (no LLM,
# no licence key, seconds); everything else runs the six-act advisor walk.
script="advisor_demo.py"
if [[ "${1:-}" == "--claims" ]]; then
    script="claims_scenario.py"
    shift
fi

echo "venv: $venv"
echo "demo: $script"
exec "$venv/bin/python" "$here/$script" "$@"
