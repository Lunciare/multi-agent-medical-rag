#!/usr/bin/env bash
# Create / refresh a self-contained venv for the validation Mini App and install
# requirements. Idempotent: safe to re-run (e.g. after a code/requirements update).
# No conda dependency — uses the system python3's bundled `venv` module.
#
# Usage:  ./deploy/setup_venv.sh
# Then run via systemd (see deploy/miniapp.service) or directly:
#   .venv/bin/uvicorn --factory backend.app:create_app --host 127.0.0.1 --port 8765
set -euo pipefail

# Repo root = parent of the directory holding this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup_venv] creating venv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "[setup_venv] venv already exists at ${VENV_DIR} (reusing)"
fi

# Use the venv's interpreter directly; no need to 'activate' for a script.
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "[setup_venv] done. uvicorn: ${VENV_DIR}/bin/uvicorn"
"${VENV_DIR}/bin/python" -c "import fastapi, uvicorn; print('[setup_venv] fastapi', fastapi.__version__, '/ uvicorn', uvicorn.__version__)"
