#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure venv exists
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "ERROR: No .venv found in project root."
    echo "Create it first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -e ."
    exit 1
fi

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"

# Run from project root
cd "$PROJECT_ROOT"

python -m safe_rl_gridworld.train "$@"
