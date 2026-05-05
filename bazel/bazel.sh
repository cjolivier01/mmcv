#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../.bazel_setup.sh"

BACKEND="$(extract_backend_arg "$@")"
BAZEL_BIN="$(pick_bazel_binary)"

if [[ -z "${MMCV_PYTHON:-}" ]]; then
  MMCV_PYTHON="$(resolve_python_for_backend "${BACKEND}")"
fi

export MMCV_PYTHON
export MMCV_BAZEL_BACKEND="${BACKEND}"

EXTRA_FLAGS=(
  "--action_env=MMCV_PYTHON=${MMCV_PYTHON}"
  "--action_env=MMCV_BAZEL_BACKEND=${MMCV_BAZEL_BACKEND}"
  "--action_env=PYTHON_BIN_PATH=${MMCV_PYTHON}"
)

args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--" ]]; then
    before=("${args[@]:0:i}")
    after=("${args[@]:i+1}")
    exec "${BAZEL_BIN}" "${before[@]}" "${EXTRA_FLAGS[@]}" -- "${after[@]}"
  fi
done

exec "${BAZEL_BIN}" "${args[@]}" "${EXTRA_FLAGS[@]}"
