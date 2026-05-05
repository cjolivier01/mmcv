from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile
from types import SimpleNamespace
import sys

PROJECT_FILES = (
    "LICENSE",
    "LICENSES.md",
    "MANIFEST.in",
    "README.md",
    "README_zh-CN.md",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
)

PROJECT_DIRS = (
    "mmcv",
    "requirements",
)

COPY_IGNORE_PATTERNS = (
    "__pycache__",
    "*.egg-info",
    "*.pyc",
    "*.pyo",
    "*.so",
    ".pytest_cache",
)

VULKAN_MARKER = "mmcv_vulkan_build.txt"
CUDA_ARCH_LIST_FALLBACK = "7.5;8.0;9.0;10.0;11.0;12.0+PTX"
TORCH_VALID_CUDA_ARCHES = {
    "3.5",
    "3.7",
    "5.0",
    "5.2",
    "5.3",
    "6.0",
    "6.1",
    "6.2",
    "7.0",
    "7.2",
    "7.5",
    "8.0",
    "8.6",
    "8.7",
    "8.9",
    "9.0",
    "9.0a",
    "10.0",
    "10.0a",
    "10.3",
    "10.3a",
    "11.0",
    "11.0a",
    "12.0",
    "12.0a",
    "12.1",
    "12.1a",
}


def repo_root() -> Path:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace:
        return Path(workspace).resolve()
    return Path(__file__).resolve().parents[1]


def detect_torch_backend() -> str:
    import torch

    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def can_build_backend(backend: str) -> bool:
    if backend == "cuda":
        try:
            from torch.utils.cpp_extension import CUDA_HOME
        except ImportError:
            return False
        return CUDA_HOME is not None or shutil.which("nvcc") is not None
    if backend == "hip":
        try:
            from torch.utils.cpp_extension import ROCM_HOME
        except ImportError:
            return False
        return ROCM_HOME is not None or shutil.which("hipcc") is not None
    if backend in {"cpu", "vulkan"}:
        return True
    return False


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested

    detected = detect_torch_backend()
    if detected in {"cuda", "hip"} and can_build_backend(detected):
        return detected
    return "cpu"


def validate_backend_python(backend: str) -> None:
    if backend in {"cpu", "vulkan"}:
        return

    detected = detect_torch_backend()
    if backend == "cuda" and detected != "cuda":
        raise RuntimeError(
            "The selected interpreter is not a CUDA PyTorch build. "
            f"Detected backend: {detected}."
        )
    if backend == "hip" and detected != "hip":
        raise RuntimeError(
            "The selected interpreter is not a ROCm/HIP PyTorch build. "
            f"Detected backend: {detected}."
        )


def _prepend_path_list(existing: str | None, additions: list[Path]) -> str:
    items = [str(path) for path in additions]
    if existing:
        items.append(existing)
    return os.pathsep.join(items)


def detect_hip_include_paths() -> list[Path]:
    candidates: list[Path] = []
    try:
        from torch.utils.cpp_extension import ROCM_HOME
    except ImportError:
        ROCM_HOME = None

    for root in (
        os.environ.get("ROCM_HOME"),
        os.environ.get("ROCM_PATH"),
        ROCM_HOME,
        "/opt/rocm",
    ):
        if root:
            candidates.append(Path(root) / "include")

    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in (Path(sys.prefix), Path(sys.base_prefix)):
        candidates.append(prefix / "include")
        candidates.append(prefix / "lib" / version_dir / "site-packages" / "_rocm_sdk_devel" / "include")

    existing: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if (resolved / "thrust" / "complex.h").is_file():
            existing.append(resolved)
    return existing


def build_env_for_backend(backend: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["MMCV_BUILD_BACKEND"] = backend
    if backend == "cpu":
        env["MMCV_WITH_OPS"] = "1"
        env.pop("FORCE_CUDA", None)
    elif backend == "cuda":
        env["MMCV_WITH_OPS"] = "1"
        env["FORCE_CUDA"] = "1"
        env.setdefault("TORCH_CUDA_ARCH_LIST", detect_torch_cuda_arch_list())
    elif backend == "hip":
        env["MMCV_WITH_OPS"] = "1"
        env.pop("FORCE_CUDA", None)
        hip_includes = detect_hip_include_paths()
        if hip_includes:
            env["CPATH"] = _prepend_path_list(env.get("CPATH"), hip_includes)
            env["CPLUS_INCLUDE_PATH"] = _prepend_path_list(
                env.get("CPLUS_INCLUDE_PATH"),
                hip_includes,
            )
    elif backend == "vulkan":
        env["MMCV_WITH_OPS"] = "0"
        env.pop("FORCE_CUDA", None)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return env


def artifact_dir(kind: str, backend: str, *, root: Path | None = None) -> Path:
    base = root or repo_root()
    if kind == "wheel":
        return base / "dist" / backend
    if kind == "build":
        return base / "build" / "bazel" / backend
    raise ValueError(f"Unsupported artifact kind: {kind}")


def stage_project_tree(source_root: Path, stage_root: Path) -> None:
    stage_root.mkdir(parents=True, exist_ok=True)
    for name in PROJECT_FILES:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, stage_root / name)

    for name in PROJECT_DIRS:
        src = source_root / name
        if src.is_dir():
            shutil.copytree(
                src,
                stage_root / name,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*COPY_IGNORE_PATTERNS),
            )


def overlay_hipified_sources(
    stage_root: Path,
    hipify_result: dict[str, SimpleNamespace],
) -> list[str]:
    written: list[str] = []
    for original_path, result in hipify_result.items():
        hipified_path = getattr(result, "hipified_path", None)
        if not hipified_path:
            continue
        src = Path(hipified_path)
        dest = Path(original_path)
        if src == dest or not src.is_file():
            continue
        try:
            dest_rel = dest.relative_to(stage_root)
        except ValueError:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        written.append(dest_rel.as_posix())

    return written


def hipify_stage(stage_root: Path) -> list[str]:
    from torch.utils.hipify import hipify_python

    hip_root = stage_root.parent / f"{stage_root.name}_hipify"
    if hip_root.exists():
        shutil.rmtree(hip_root)

    hipify_result = hipify_python.hipify(
        project_directory=str(stage_root),
        output_directory=str(hip_root),
        includes=["mmcv/ops/csrc/*", "mmcv/ops/csrc/**/*"],
        show_progress=False,
        is_pytorch_extension=True,
    )
    return overlay_hipified_sources(stage_root, hipify_result)


def locate_built_extension(stage_root: Path) -> Path:
    matches = sorted((stage_root / "mmcv").glob("_ext*.so"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one built mmcv._ext artifact, found {len(matches)} in "
            f"{stage_root / 'mmcv'}."
        )
    return matches[0]


def newest_wheel(output_dir: Path) -> Path:
    wheels = sorted(output_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"No wheel was created in {output_dir}.")
    return wheels[-1]


def wheel_has_extension(wheel_path: Path) -> bool:
    with ZipFile(wheel_path) as archive:
        return any(
            member.startswith("mmcv/_ext") and member.endswith(".so")
            for member in archive.namelist()
        )


def clear_matching(output_dir: Path, patterns: list[str]) -> None:
    if not output_dir.exists():
        return

    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def write_vulkan_marker(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / VULKAN_MARKER
    marker.write_text(
        "Vulkan targets currently package mmcv-lite without mmcv._ext because "
        "this tree does not contain native Vulkan custom-op kernels.\n",
        encoding="utf-8",
    )
    return marker


def _sm_value_to_arch(sm_value: int) -> str:
    major = sm_value // 10
    minor = sm_value % 10
    if sm_value >= 100:
        major = sm_value // 10
        minor = sm_value % 10
        return f"{major}.{minor}"
    return f"{major}.{minor}"


def _cuda_arch_sort_key(arch: str) -> tuple[int, int]:
    base = arch.removesuffix("+PTX").removesuffix("a")
    major_str, minor_str = base.split(".", 1)
    return int(major_str), int(minor_str)


def compact_cuda_arch_list(arches: list[str]) -> list[str]:
    compact_by_major: dict[int, str] = {}
    for arch in sorted(set(arches), key=_cuda_arch_sort_key):
        major, _minor = _cuda_arch_sort_key(arch)
        compact_by_major.setdefault(major, arch)
    return [compact_by_major[major] for major in sorted(compact_by_major)]


def detect_torch_cuda_arch_list() -> str:
    override = os.environ.get("MMCV_TORCH_CUDA_ARCH_LIST")
    if override:
        return override

    existing = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if existing:
        return existing

    try:
        result = subprocess.run(
            ["nvcc", "--list-gpu-code"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return CUDA_ARCH_LIST_FALLBACK
    if result.returncode != 0:
        return CUDA_ARCH_LIST_FALLBACK

    arch_values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("sm_"):
            continue
        try:
            arch_values.append(int(line.split("_", 1)[1]))
        except ValueError:
            continue

    if not arch_values:
        return CUDA_ARCH_LIST_FALLBACK

    unique_values = sorted(set(arch_values))
    arch_list = compact_cuda_arch_list([
        _sm_value_to_arch(value)
        for value in unique_values
        if _sm_value_to_arch(value) in TORCH_VALID_CUDA_ARCHES
    ])
    if not arch_list:
        return CUDA_ARCH_LIST_FALLBACK
    arch_list[-1] = f"{arch_list[-1]}+PTX"
    return ";".join(arch_list)
