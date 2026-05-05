#!/bin/bash
set -euo pipefail

WORKSPACE_DIR="${BUILD_WORKSPACE_DIRECTORY:-}"
if [[ -z "${WORKSPACE_DIR}" ]]; then
  WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

PYTHON_BIN="${MMCV_PYTHON:-${PYTHON_BIN_PATH:-}}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export PYTHONPATH="${WORKSPACE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${WORKSPACE_DIR}/build_tools/bazel_cli.py" "$@"
