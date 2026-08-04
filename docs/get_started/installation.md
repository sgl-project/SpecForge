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

For the pinned ROCm environment, install the checked-in requirements before the
package:

```bash
python -m pip install -r requirements-rocm.txt
python -m pip install -e .
```

The file pins a ROCm 7.2 PyTorch stack. Use a wheel index and driver combination
compatible with the host if your ROCm version differs. Online runs require a
ROCm-compatible SGLang capture service; offline feature consumers can start
without target inference. PyTorch exposes ROCm accelerators through its
`torch.cuda` API and uses NCCL for distributed runs.

If your ROCm `torch` build is pulled from a separate index, install SpecForge
without dependencies (`pip install --no-deps -e .`) so the ROCm wheel is not
silently replaced by the CUDA `torch` pinned in `pyproject.toml`.

Known issues on ROCm:

- `ImportError: cannot import name 'VideoReader' from 'torchvision.io'` — newer
  `datasets` tensorizes through torchvision video support that some ROCm
  torchvision builds omit. Install a torchvision build with video support or pin
  an older `datasets`.
- Gated target models (e.g. `meta-llama/*`) require a Hugging Face token;
  otherwise embedding loading fails with `403 Forbidden`. Log in with
  `huggingface-cli login`, or use an already-cached model with `HF_HUB_OFFLINE=1`.
- Harmless `torch._dynamo` "compilation metrics ... not JSON serializable"
  warnings and FSDP deprecation `FutureWarning`s may appear during training; they
  do not affect the run.

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
