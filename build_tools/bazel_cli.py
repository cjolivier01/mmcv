from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from build_tools import bazel_support


@contextmanager
def staged_project(backend: str, *, hipify: bool, keep_stage: bool):
    source_root = bazel_support.repo_root()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"mmcv-{backend}-"))
    stage_root = temp_dir / "src"
    bazel_support.stage_project_tree(source_root, stage_root)

    hipified = []
    if backend == "hip" and hipify:
        hipified = bazel_support.hipify_stage(stage_root)

    try:
        yield stage_root, hipified
    finally:
        if keep_stage:
            print(f"Kept staged tree at {stage_root}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_setup(stage_root: Path, env: dict[str, str], setup_args: list[str]) -> None:
    subprocess.run(
        [sys.executable, "setup.py", *setup_args],
        cwd=stage_root,
        env=env,
        check=True,
    )


def build_extension(args: argparse.Namespace) -> int:
    backend = bazel_support.resolve_backend(args.backend)
    bazel_support.validate_backend_python(backend)
    output_dir = Path(args.output_dir) if args.output_dir else bazel_support.artifact_dir("build", backend)
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "vulkan":
        marker = bazel_support.write_vulkan_marker(output_dir)
        print(f"Wrote {marker}")
        return 0

    bazel_support.clear_matching(output_dir, ["_ext*.so"])
    env = bazel_support.build_env_for_backend(backend)
    with staged_project(backend, hipify=not args.no_hipify, keep_stage=args.keep_stage) as (stage_root, hipified):
        if hipified:
            print(f"Applied HIP overlay to {len(hipified)} source files.")
        run_setup(stage_root, env, ["build_ext", "--inplace"])
        extension_path = bazel_support.locate_built_extension(stage_root)
        out_path = output_dir / extension_path.name
        shutil.copy2(extension_path, out_path)
        print(f"Wrote {out_path}")
    return 0


def build_wheel(args: argparse.Namespace) -> int:
    backend = bazel_support.resolve_backend(args.backend)
    bazel_support.validate_backend_python(backend)
    output_dir = Path(args.output_dir) if args.output_dir else bazel_support.artifact_dir("wheel", backend)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = bazel_support.build_env_for_backend(backend)
    before = {path.name for path in output_dir.glob("*.whl")}
    with staged_project(backend, hipify=not args.no_hipify, keep_stage=args.keep_stage) as (stage_root, hipified):
        if backend == "vulkan":
            print(
                "Building a Vulkan-compatible wheel by packaging mmcv-lite "
                "without custom ops."
            )
        else:
            if hipified:
                print(f"Applied HIP overlay to {len(hipified)} source files.")
            run_setup(stage_root, env, ["build_ext", "--inplace"])
        run_setup(
            stage_root,
            env,
            ["bdist_wheel", "--dist-dir", str(output_dir), "--skip-build"],
        )

    after = {path.name for path in output_dir.glob("*.whl")}
    created = sorted(after - before)
    wheel_paths = [output_dir / wheel_name for wheel_name in created]
    if not wheel_paths:
        wheel_paths = [bazel_support.newest_wheel(output_dir)]

    expected_ops = backend != "vulkan"
    for wheel_path in wheel_paths:
        found_ops = bazel_support.wheel_has_extension(wheel_path)
        if found_ops != expected_ops:
            raise RuntimeError(
                f"Wheel {wheel_path} has extension={found_ops}, expected {expected_ops} "
                f"for backend {backend}."
            )
        print(f"Wrote {wheel_path}")
    return 0


def develop_extension(args: argparse.Namespace) -> int:
    backend = bazel_support.resolve_backend(args.backend)
    bazel_support.validate_backend_python(backend)
    workspace_root = bazel_support.repo_root()

    if backend == "vulkan":
        dest_dir = workspace_root / "mmcv"
        bazel_support.clear_matching(dest_dir, ["_ext*.so"])
        marker = bazel_support.write_vulkan_marker(bazel_support.artifact_dir("build", backend, root=workspace_root))
        print(
            "Vulkan develop mode does not copy mmcv._ext because no native "
            f"Vulkan extension exists. Wrote {marker} instead."
        )
        return 0

    env = bazel_support.build_env_for_backend(backend)
    with staged_project(backend, hipify=not args.no_hipify, keep_stage=args.keep_stage) as (stage_root, hipified):
        if hipified:
            print(f"Applied HIP overlay to {len(hipified)} source files.")
        run_setup(stage_root, env, ["build_ext", "--inplace"])
        extension_path = bazel_support.locate_built_extension(stage_root)
        dest_dir = workspace_root / "mmcv"
        bazel_support.clear_matching(dest_dir, ["_ext*.so"])
        dest_path = dest_dir / extension_path.name
        shutil.copy2(extension_path, dest_path)
        print(f"Linked {dest_path}")
    return 0


def smoke_import(args: argparse.Namespace) -> int:
    backend = bazel_support.resolve_backend(args.backend)
    bazel_support.validate_backend_python(backend)
    env = bazel_support.build_env_for_backend(backend)

    with staged_project(backend, hipify=not args.no_hipify, keep_stage=args.keep_stage) as (stage_root, hipified):
        if backend in {"cuda", "hip"}:
            if hipified:
                print(f"Applied HIP overlay to {len(hipified)} source files.")
            run_setup(stage_root, env, ["build_ext", "--inplace"])

        smoke_env = dict(env)
        smoke_env["PYTHONPATH"] = f"{stage_root}{os.pathsep}{smoke_env.get('PYTHONPATH', '')}"
        smoke_code = """
import importlib.util
from pathlib import Path
import sys
import torch

backend = sys.argv[1]
stage_root = Path(sys.argv[2])
extensions = sorted((stage_root / "mmcv").glob("_ext*.so"))
expected_ops = backend != "vulkan"
found_ops = bool(extensions)
assert found_ops == expected_ops, (backend, found_ops, expected_ops)
if expected_ops:
    spec = importlib.util.spec_from_file_location("mmcv._ext", extensions[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
print(torch.__version__, backend, found_ops)
"""
        subprocess.run(
            [sys.executable, "-c", smoke_code, backend, str(stage_root)],
            cwd=stage_root,
            env=smoke_env,
            check=True,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bazel-driven MMCV build entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--backend",
            choices=("auto", "cpu", "cuda", "hip", "vulkan"),
            default="auto",
        )
        subparser.add_argument("--output-dir")
        subparser.add_argument("--keep-stage", action="store_true")
        subparser.add_argument("--no-hipify", action="store_true")

    build_ext_parser = subparsers.add_parser("build-ext")
    add_common_flags(build_ext_parser)
    build_ext_parser.set_defaults(func=build_extension)

    wheel_parser = subparsers.add_parser("wheel")
    add_common_flags(wheel_parser)
    wheel_parser.set_defaults(func=build_wheel)

    develop_parser = subparsers.add_parser("develop")
    add_common_flags(develop_parser)
    develop_parser.set_defaults(func=develop_extension)

    smoke_parser = subparsers.add_parser("smoke-import")
    add_common_flags(smoke_parser)
    smoke_parser.set_defaults(func=smoke_import)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
