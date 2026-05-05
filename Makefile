BAZEL := bazel/bazel.sh

all: print_targets

.PHONY: all build build-cpu build-cuda build-hip build-vulkan clean develop \
	develop-cpu develop-cuda develop-hip distclean expunge print_targets test \
	test-cpu test-cuda test-hip test-vulkan wheel wheel-cpu wheel-cuda \
	wheel-hip wheel-vulkan

build:
	$(BAZEL) run --config=release //:build_ext

build-cpu:
	MMCV_PYTHON="$${MMCV_CPU_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cpu //:build_ext_cpu

build-cuda:
	MMCV_PYTHON="$${MMCV_CUDA_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cuda //:build_ext_cuda

build-hip:
	MMCV_PYTHON="$${MMCV_HIP_PYTHON:-}" $(BAZEL) run --config=release --define=backend=hip //:build_ext_hip

build-vulkan:
	MMCV_PYTHON="$${MMCV_VULKAN_PYTHON:-}" $(BAZEL) run --config=release --define=backend=vulkan //:build_ext_vulkan

wheel:
	$(BAZEL) run --config=release //:bdist_wheel

wheel-cpu:
	MMCV_PYTHON="$${MMCV_CPU_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cpu //:bdist_wheel_cpu

wheel-cuda:
	MMCV_PYTHON="$${MMCV_CUDA_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cuda //:bdist_wheel_cuda

wheel-hip:
	MMCV_PYTHON="$${MMCV_HIP_PYTHON:-}" $(BAZEL) run --config=release --define=backend=hip //:bdist_wheel_hip

wheel-vulkan:
	MMCV_PYTHON="$${MMCV_VULKAN_PYTHON:-}" $(BAZEL) run --config=release --define=backend=vulkan //:bdist_wheel_vulkan

develop:
	$(BAZEL) run --config=release //:develop

develop-cpu:
	MMCV_PYTHON="$${MMCV_CPU_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cpu //:develop_cpu

develop-cuda:
	MMCV_PYTHON="$${MMCV_CUDA_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cuda //:develop_cuda

develop-hip:
	MMCV_PYTHON="$${MMCV_HIP_PYTHON:-}" $(BAZEL) run --config=release --define=backend=hip //:develop_hip

test:
	$(BAZEL) test --config=release //:build_tools_test
	$(BAZEL) run --config=release //:smoke_import

test-cpu:
	MMCV_PYTHON="$${MMCV_CPU_PYTHON:-}" $(BAZEL) test --config=release //:build_tools_test
	MMCV_PYTHON="$${MMCV_CPU_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cpu //:smoke_import_cpu

test-cuda:
	MMCV_PYTHON="$${MMCV_CUDA_PYTHON:-}" $(BAZEL) test --config=release //:build_tools_test
	MMCV_PYTHON="$${MMCV_CUDA_PYTHON:-}" $(BAZEL) run --config=release --define=backend=cuda //:smoke_import_cuda

test-hip:
	MMCV_PYTHON="$${MMCV_HIP_PYTHON:-}" $(BAZEL) test --config=release //:build_tools_test
	MMCV_PYTHON="$${MMCV_HIP_PYTHON:-}" $(BAZEL) run --config=release --define=backend=hip //:smoke_import_hip

test-vulkan:
	MMCV_PYTHON="$${MMCV_VULKAN_PYTHON:-}" $(BAZEL) test --config=release //:build_tools_test
	MMCV_PYTHON="$${MMCV_VULKAN_PYTHON:-}" $(BAZEL) run --config=release --define=backend=vulkan //:smoke_import_vulkan

clean:
	$(BAZEL) clean

distclean expunge:
	$(BAZEL) clean --expunge

print_targets:
	@printf '%s\n' \
		"Available make targets (run 'make <target>'):" \
		'' \
		'Primary workflow' \
		'----------------' \
		'build         Builds the extension for the currently selected backend.' \
		'wheel         Builds a wheel for the currently selected backend and verifies expected wheel contents.' \
		'develop       Builds the extension and copies mmcv._ext into the workspace package.' \
		'test          Runs Bazel unit tests and a direct extension smoke load for the current backend.' \
		'' \
		'Explicit backends' \
		'-----------------' \
		'build-cpu     Builds the CPU extension using MMCV_CPU_PYTHON or the detected Python interpreter.' \
		'build-cuda    Builds the CUDA extension using MMCV_CUDA_PYTHON or the detected CUDA interpreter.' \
		'build-hip     Builds the ROCm/HIP extension using MMCV_HIP_PYTHON or the detected ROCm interpreter.' \
		'build-vulkan  Produces the Vulkan marker target; Vulkan wheels currently package mmcv-lite without custom ops.' \
		'wheel-cpu     Builds the CPU wheel into dist/cpu/.' \
		'wheel-cuda    Builds the CUDA wheel into dist/cuda/.' \
		'wheel-hip     Builds the ROCm/HIP wheel into dist/hip/.' \
		'wheel-vulkan  Builds the Vulkan-compatible mmcv-lite wheel into dist/vulkan/.' \
		'develop-cpu   Copies a CPU-built mmcv._ext into the workspace package.' \
		'develop-cuda  Copies a CUDA-built mmcv._ext into the workspace package.' \
		'develop-hip   Copies a HIP-built mmcv._ext into the workspace package.' \
		'test-cpu      Runs helper tests plus the CPU extension smoke build/load flow.' \
		'test-cuda     Runs helper tests plus the CUDA smoke build/import flow.' \
		'test-hip      Runs helper tests plus the HIP smoke build/import flow.' \
		'test-vulkan   Runs helper tests plus the Vulkan packaging marker smoke flow.' \
		'' \
		'Useful environment variables' \
		'----------------------------' \
		'MMCV_TORCH_CUDA_ARCH_LIST  Overrides the CUDA arch list passed through to PyTorch extension builds.' \
		'PYTORCH_ROCM_ARCH          Narrows HIP/ROCm codegen to specific gfx targets during wheel builds.' \
		'MAX_JOBS                   Overrides PyTorch extension parallelism for both CUDA and HIP builds.' \
		'' \
		'Maintenance' \
		'-----------' \
		'clean        Runs bazel clean.' \
		'distclean    Runs bazel clean --expunge.' \
		'expunge      Alias for distclean.' \
		'print_targets  Shows this help text.'
