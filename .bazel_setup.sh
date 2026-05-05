pick_bazel_binary() {
  if command -v bazelisk >/dev/null 2>&1; then
    command -v bazelisk
    return 0
  fi

  command -v bazel
}

conda_base_path() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base 2>/dev/null || true
  fi
}

first_existing_python() {
  local candidate
  for candidate in "$@"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

resolve_python_for_backend() {
  local backend="${1:-auto}"
  local conda_base

  if [ -n "${MMCV_PYTHON:-}" ]; then
    printf '%s\n' "${MMCV_PYTHON}"
    return 0
  fi

  conda_base="$(conda_base_path)"

  case "${backend}" in
    hip)
      first_existing_python \
        "${MMCV_HIP_PYTHON:-}" \
        "${CONDA_PREFIX:-}/bin/python" \
        "${conda_base}/envs/ubuntu-rocm/bin/python" \
        "${conda_base}/envs/rocm/bin/python" \
        "$(command -v python 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)"
      ;;
    cuda)
      first_existing_python \
        "${MMCV_CUDA_PYTHON:-}" \
        "${CONDA_PREFIX:-}/bin/python" \
        "${conda_base:+${conda_base}/bin/python}" \
        "$(command -v python 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)"
      ;;
    vulkan|auto|*)
      first_existing_python \
        "${MMCV_VULKAN_PYTHON:-}" \
        "$(command -v python 2>/dev/null || true)" \
        "$(command -v python3 2>/dev/null || true)"
      ;;
  esac
}

extract_backend_arg() {
  local arg

  for arg in "$@"; do
    case "${arg}" in
      --define=backend=*)
        printf '%s\n' "${arg#--define=backend=}"
        return 0
        ;;
      *_cpu)
        printf '%s\n' "cpu"
        return 0
        ;;
      *_cuda)
        printf '%s\n' "cuda"
        return 0
        ;;
      *_hip)
        printf '%s\n' "hip"
        return 0
        ;;
      *_vulkan)
        printf '%s\n' "vulkan"
        return 0
        ;;
    esac
  done

  printf '%s\n' "${MMCV_BAZEL_BACKEND:-auto}"
}
