# Get Started

## Installation

To install this project, you can simply run the following command.

- **Install from source (recommended)**

```bash
# git clone the source code
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

# create a new virtual environment
uv venv -p 3.11 --seed
source .venv/bin/activate

# Install CUDA 13 training/server dependencies, then SpecForge and its plugin.
# Building the pinned SGLang revision requires Rust 1.92, C++ and CUDA development tools.
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install wheel setuptools setuptools-rust setuptools-scm
uv pip install -v -e . --no-build-isolation --prerelease=allow
# Install SGLang explicitly; SpecForge's package dependencies do not install it.
uv pip install --no-build-isolation --prerelease=allow \
  "sglang @ git+https://github.com/sgl-project/sglang.git@c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0#subdirectory=python"
uv pip install --no-deps plugins/sglang-spec-capture
uv pip install mooncake-transfer-engine-cuda13
```

Install SGLang commit `c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0` manually
using the command above for capture workloads. SpecForge pins Torch `2.14.1`
and Transformers `5.19.0`, but does not install SGLang automatically.
Normal installer caches are reused. Online
training and offline feature preparation use the capture plugin; see the
[plugin guide](../../../plugins/sglang-spec-capture/README.md) for server setup.
Offline training from existing features does not require a capture server.

- **Install from PyPI**

::: code-group

```bash [uv]
uv pip install specforge --prerelease=allow
```

```bash [pip]
pip install specforge
```

:::

PyPI installation also requires the explicit SGLang installation above when
using a capture server. Install SGLang and the capture plugin in the server's
environment; an existing compatible server can be used separately.

## Accelerator-specific environments

### NVIDIA CUDA

The standard installation above uses the platform selected by PyTorch. Install
a CUDA build compatible with the host driver, then run every recipe through
the same `specforge train` entry. On Linux it also installs `liger-kernel`,
whose RMSNorm/SwiGLU kernels DFlash drafts use by default on CUDA and ROCm
(`model.use_liger_kernel`).

### AMD ROCm

On ROCm, install SpecForge into an environment that already provides a ROCm
PyTorch and a ROCm SGLang (an official SGLang ROCm release container is the
recommended base), and install the package **without dependencies** so pip does
not pull CUDA wheels over the working ROCm stack:

```bash
# Inside the ROCm SGLang container
git clone https://github.com/sgl-project/SpecForge.git /workspace/SpecForge
cd /workspace/SpecForge
python -m pip install -e . --no-deps
# DFlash drafts use Liger RMSNorm/SwiGLU by default on ROCm; --no-deps keeps
# the container's ROCm Triton.
python -m pip install --no-deps "liger-kernel>=0.8.3,<0.9"
```

For the complete container setup and an end-to-end walkthrough covering
installation, data preparation, offline colocated training, online
disaggregated training, and its single-supervisor and split external launch
forms on AMD Instinct GPUs, follow the
[AMD ROCm Tutorial](../basic_usage/AMD/amd_rocm.md).

### Ascend NPU

Install the vendor-matched PyTorch and `torch_npu` packages first, then install
SpecForge. The checked-in
[`qwen3.5-4b-dflash-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml)
and
[`qwen3.5-4b-domino-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-domino-npu.yaml)
recipes use external SGLang server capture with SDPA consumers. Install a
compatible SGLang/Mooncake service first. The unified launcher detects the NPU
device, self-launches the process count recorded in YAML, and selects HCCL; see
the [training guide](../basic_usage/training.md#cuda-rocm-and-ascend-npu).
DFlash drafts keep the native RMSNorm/SwiGLU on NPU; Liger's Ascend backend is
not validated in SpecForge.

### Intel GPU

On XPU, install SpecForge into an environment that already provides a XPU
PyTorch and a XPU SGLang (an official SGLang XPU release container is the
recommended base), and install the packages:

```bash
# Inside the XPU SGLang container
git clone https://github.com/sgl-project/SpecForge.git /sgl-workspace/SpecForge
cd /sgl-workspace/SpecForge
cp pyproject.toml pyproject.toml.backup
cp pyproject_xpu.toml pyproject.toml
python -m pip install -e .
```
