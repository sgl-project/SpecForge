# 🚀 Get Started

## 📦 Installation

To install this project, you can simply run the following command.

- **Install from source (recommended)**

```bash
# git clone the source code
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

# create a new virtual environment
uv venv -p 3.11
source .venv/bin/activate

# install specforge
uv pip install -v . --prerelease=allow
```

- **Install from PyPI**

```bash
pip install specforge
```

## Accelerator-specific environments

### NVIDIA CUDA

The standard installation above uses the platform selected by PyTorch. Install
a CUDA build compatible with the host driver, then run every recipe through
the same `specforge train` entry.

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
```

For the complete container setup and an end-to-end walkthrough covering
installation, data preparation, and offline / online / disaggregated training on
AMD Instinct GPUs, follow the [AMD ROCm Tutorial](../basic_usage/AMD/amd_rocm.md).

### Ascend NPU

Install the vendor-matched PyTorch and `torch_npu` packages first, then install
SpecForge. The checked-in
[`qwen3.5-4b-dflash-online-npu.yaml`](../../examples/configs/qwen3.5-4b-dflash-online-npu.yaml)
and
[`qwen3.5-4b-domino-online-npu.yaml`](../../examples/configs/qwen3.5-4b-domino-online-npu.yaml)
recipes use external SGLang server capture with SDPA consumers. Install a
compatible SGLang/Mooncake service first. The unified launcher detects the NPU
device, self-launches the process count recorded in YAML, and selects HCCL; see
the [training guide](../basic_usage/training.md#cuda-rocm-and-ascend-npu).
