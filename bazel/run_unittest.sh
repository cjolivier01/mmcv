#!/bin/bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" && -n "${TEST_WORKSPACE:-}" ]]; then
  ROOT="${TEST_SRCDIR}/${TEST_WORKSPACE}"
elif [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
  ROOT="${BUILD_WORKSPACE_DIRECTORY}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -f "${ROOT}/openmm/mmcv/tests/test_build_tools.py" ]]; then
  ROOT="${ROOT}/openmm/mmcv"
fi
export BUILD_WORKSPACE_DIRECTORY="${ROOT}"

PYTHON_BIN="${MMCV_PYTHON:-${PYTHON_BIN_PATH:-}}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" tests/test_build_tools.py
