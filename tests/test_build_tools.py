from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from build_tools import bazel_cli, bazel_support


class BazelSupportTest(unittest.TestCase):

    def test_build_env_for_backend_sets_expected_flags(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(
            bazel_support,
            "detect_hip_include_paths",
            return_value=[Path("/opt/rocm/include"), Path("/tmp/rocm-sdk/include")],
        ):
            cpu_env = bazel_support.build_env_for_backend("cpu")
            cuda_env = bazel_support.build_env_for_backend("cuda")
            hip_env = bazel_support.build_env_for_backend("hip")
            vulkan_env = bazel_support.build_env_for_backend("vulkan")

        self.assertEqual(cpu_env["MMCV_WITH_OPS"], "1")
        self.assertNotIn("FORCE_CUDA", cpu_env)
        self.assertEqual(cuda_env["MMCV_WITH_OPS"], "1")
        self.assertEqual(cuda_env["FORCE_CUDA"], "1")
        self.assertEqual(hip_env["MMCV_WITH_OPS"], "1")
        self.assertNotIn("FORCE_CUDA", hip_env)
        self.assertEqual(
            hip_env["CPATH"],
            "/opt/rocm/include:/tmp/rocm-sdk/include",
        )
        self.assertEqual(
            hip_env["CPLUS_INCLUDE_PATH"],
            "/opt/rocm/include:/tmp/rocm-sdk/include",
        )
        self.assertEqual(vulkan_env["MMCV_WITH_OPS"], "0")
        self.assertNotIn("MAX_JOBS", cuda_env)

    def test_resolve_backend_uses_cpu_for_cpu_only_python(self) -> None:
        with mock.patch.object(
            bazel_support, "detect_torch_backend", return_value="cpu"
        ):
            self.assertEqual(bazel_support.resolve_backend("auto"), "cpu")

    def test_resolve_backend_falls_back_to_cpu_when_accelerator_toolchain_is_missing(
        self,
    ) -> None:
        with mock.patch.object(
            bazel_support, "detect_torch_backend", return_value="cuda"
        ), mock.patch.object(bazel_support, "can_build_backend", return_value=False):
            self.assertEqual(bazel_support.resolve_backend("auto"), "cpu")

        with mock.patch.object(
            bazel_support, "detect_torch_backend", return_value="hip"
        ), mock.patch.object(bazel_support, "can_build_backend", return_value=True):
            self.assertEqual(bazel_support.resolve_backend("auto"), "hip")

    def test_overlay_hipified_sources_uses_hipify_result_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            stage_root = temp_root / "stage"
            hip_root = temp_root / "hipified"

            original_kernel = stage_root / "mmcv/ops/csrc/pytorch/cuda/example_cuda.cu"
            original_header = (
                stage_root / "mmcv/ops/csrc/common/cuda/pytorch_cuda_helper.hpp"
            )
            original_info = stage_root / "mmcv/ops/csrc/pytorch/info.cpp"
            for path in (original_kernel, original_header, original_info):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("// original\n", encoding="utf-8")

            hip_kernel = hip_root / "mmcv/ops/csrc/pytorch/hip/example_hip.hip"
            hip_header = hip_root / "mmcv/ops/csrc/common/hip/pytorch_hip_helper.hpp"
            hip_info = hip_root / "mmcv/ops/csrc/pytorch/info_hip.cpp"
            hip_kernel.parent.mkdir(parents=True, exist_ok=True)
            hip_header.parent.mkdir(parents=True, exist_ok=True)
            hip_info.parent.mkdir(parents=True, exist_ok=True)
            hip_kernel.write_text("// hip kernel\n", encoding="utf-8")
            hip_header.write_text("// hip header\n", encoding="utf-8")
            hip_info.write_text("// hip info\n", encoding="utf-8")

            written = bazel_support.overlay_hipified_sources(
                stage_root,
                {
                    str(original_kernel): SimpleNamespace(
                        hipified_path=str(hip_kernel)
                    ),
                    str(original_header): SimpleNamespace(
                        hipified_path=str(hip_header)
                    ),
                    str(original_info): SimpleNamespace(hipified_path=str(hip_info)),
                },
            )

            self.assertIn("mmcv/ops/csrc/pytorch/cuda/example_cuda.cu", written)
            self.assertIn("mmcv/ops/csrc/common/cuda/pytorch_cuda_helper.hpp", written)
            self.assertIn("mmcv/ops/csrc/pytorch/info.cpp", written)
            self.assertEqual(
                original_kernel.read_text(encoding="utf-8"),
                "// hip kernel\n",
            )
            self.assertEqual(
                original_header.read_text(encoding="utf-8"),
                "// hip header\n",
            )
            self.assertEqual(original_info.read_text(encoding="utf-8"), "// hip info\n")

    def test_stage_project_tree_copies_required_files_and_skips_built_extensions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            stage_root = temp_root / "stage"
            (source_root / "mmcv").mkdir(parents=True)
            (source_root / "requirements").mkdir(parents=True)
            (source_root / "setup.py").write_text("print('setup')\n", encoding="utf-8")
            (source_root / "mmcv" / "__init__.py").write_text(
                "__all__ = []\n", encoding="utf-8"
            )
            (source_root / "mmcv" / "_ext.cpython-312-x86_64-linux-gnu.so").write_text(
                "binary\n", encoding="utf-8"
            )
            (source_root / "requirements" / "runtime.txt").write_text(
                "numpy\n", encoding="utf-8"
            )

            bazel_support.stage_project_tree(source_root, stage_root)

            self.assertTrue((stage_root / "setup.py").is_file())
            self.assertTrue((stage_root / "mmcv" / "__init__.py").is_file())
            self.assertTrue((stage_root / "requirements" / "runtime.txt").is_file())
            self.assertFalse(any((stage_root / "mmcv").glob("_ext*.so")))

    def test_detect_torch_cuda_arch_list_compacts_nvcc_arches_per_major(self) -> None:
        nvcc_output = """sm_75
sm_80
sm_86
sm_89
sm_90
sm_100
sm_103
sm_110
sm_120
sm_121
"""
        with mock.patch.object(
            bazel_support.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=["nvcc", "--list-gpu-code"],
                returncode=0,
                stdout=nvcc_output,
                stderr="",
            ),
        ):
            self.assertEqual(
                bazel_support.detect_torch_cuda_arch_list(),
                "7.5;8.0;9.0;10.0;11.0;12.0+PTX",
            )

    def test_extract_backend_arg_recognizes_target_suffixes(self) -> None:
        script = """
source ./.bazel_setup.sh
extract_backend_arg //:build_ext_cuda
extract_backend_arg //:bdist_wheel_hip
extract_backend_arg //:smoke_import_vulkan
extract_backend_arg //:build_ext_cpu
"""
        result = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            cwd=bazel_support.repo_root(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip().splitlines(), ["cuda", "hip", "vulkan", "cpu"]
        )

    def test_wheel_helpers_validate_python_package_and_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            good_wheel = temp_root / "mmcv-good.whl"
            bad_wheel = temp_root / "mmcv-bad.whl"

            with zipfile.ZipFile(good_wheel, "w") as archive:
                archive.writestr("mmcv/__init__.py", "__version__ = '2.2.0'\n")
                archive.writestr("mmcv/image/__init__.py", "from .io import imread\n")
                archive.writestr("mmcv/_ext.cpython-312-x86_64-linux-gnu.so", "binary")

            with zipfile.ZipFile(bad_wheel, "w") as archive:
                archive.writestr("mmcv/_ext.cpython-312-x86_64-linux-gnu.so", "binary")

            self.assertTrue(bazel_support.wheel_has_extension(good_wheel))
            self.assertTrue(bazel_support.wheel_has_python_package(good_wheel))
            self.assertTrue(bazel_support.wheel_has_extension(bad_wheel))
            self.assertFalse(bazel_support.wheel_has_python_package(bad_wheel))

    def test_build_wheel_does_not_skip_python_package_build(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            stage_root = temp_root / "stage"
            output_dir = temp_root / "dist"
            stage_root.mkdir()
            output_dir.mkdir()
            setup_calls: list[list[str]] = []

            @contextmanager
            def fake_staged_project(*args, **kwargs):
                del args, kwargs
                yield stage_root, []

            def fake_run_setup(
                current_stage_root: Path,
                env: dict[str, str],
                setup_args: list[str],
            ) -> None:
                del current_stage_root, env
                setup_calls.append(setup_args)
                if setup_args[0] != "bdist_wheel":
                    return
                wheel_path = output_dir / "mmcv-2.2.0-cp312-cp312-linux_x86_64.whl"
                with zipfile.ZipFile(wheel_path, "w") as archive:
                    archive.writestr("mmcv/__init__.py", "__version__ = '2.2.0'\n")
                    archive.writestr(
                        "mmcv/image/__init__.py", "from .io import imread\n"
                    )
                    archive.writestr(
                        "mmcv/_ext.cpython-312-x86_64-linux-gnu.so", "binary"
                    )

            args = argparse.Namespace(
                backend="hip",
                output_dir=str(output_dir),
                no_hipify=False,
                keep_stage=False,
            )
            with mock.patch.object(
                bazel_support, "resolve_backend", return_value="hip"
            ), mock.patch.object(
                bazel_support, "validate_backend_python"
            ), mock.patch.object(
                bazel_support, "build_env_for_backend", return_value={}
            ), mock.patch.object(
                bazel_cli, "staged_project", fake_staged_project
            ), mock.patch.object(
                bazel_cli, "run_setup", side_effect=fake_run_setup
            ):
                self.assertEqual(bazel_cli.build_wheel(args), 0)

            self.assertEqual(setup_calls[0], ["build_ext", "--inplace"])
            self.assertEqual(
                setup_calls[1],
                ["bdist_wheel", "--dist-dir", str(output_dir)],
            )
            self.assertNotIn("--skip-build", setup_calls[1])


if __name__ == "__main__":
    unittest.main()
