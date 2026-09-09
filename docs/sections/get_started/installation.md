# Get Started

## Installation

SpecForge needs Python 3.10 or newer. The recommended installer is
[uv](https://docs.astral.sh/uv/): the `cuda` extra routes `torch` and
`sglang-kernel` to CUDA-specific wheel indexes through `[tool.uv.sources]` in
`pyproject.toml`, and uv is what honours those pins. pip works too; it takes
every wheel from PyPI, where the pinned `torch` and `sglang-kernel` releases
are CUDA 13 builds anyway.

### Quick install

Pick your hardware, the SpecForge version, your installer and any optional
extras. The command updates as you go.

<InstallSelector />

The selector covers the supported combinations. The sections below explain
what the generated command does on each accelerator and what to watch out for.

### Extras reference

| Hardware | Extra | What it pins |
| --- | --- | --- |
| NVIDIA GPU (CUDA 13 driver) | `cuda` | `torch==2.13.0` (cu130), `sglang-kernel==0.4.6.post1` (cu130), `mooncake-transfer-engine-cuda13` |
| AMD Instinct GPU (ROCm) | none, install with `--no-deps` | ROCm PyTorch and SGLang come from the SGLang ROCm container |
| Ascend NPU | `npu` | `torch==2.13.0+cpu` (PyTorch CPU index), `torch_npu==2.13.0rc1`, `triton==3.5.0`, `triton_ascend==3.2.0` (Python 3.11 or older); the NPU build of SGLang, `sgl_kernel_npu` and `hccl` come from the CANN stack |

SGLang is not a base dependency: the `cuda` extra pins the CUDA build
(`sglang==0.5.18`), while ROCm and NPU installs use the SGLang build that ships
with their vendor stack. Every install shares the same `specforge train` entry
point; only the compiled wheels differ. Optional extras that stack on top:

| Extra | Adds |
| --- | --- |
| `fa` | `flash-attn` (built from source; install `torch` first and build with `--no-build-isolation`) |
| `liger` | `liger-kernel`, enables `model.use_liger_kernel` for DFlash training |
| `dev` | `pre-commit` for contributors |

Combine extras with a comma, for example `".[cuda,liger]"`.

### NVIDIA CUDA

The canonical source install is:

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge

uv venv -p 3.11 --seed
source .venv/bin/activate

uv pip install --prerelease=allow -e ".[cuda]"
```

`--prerelease=allow` (`--pre` for pip) is required because SGLang 0.5.18 pins
a pre-release `cuda-tile` wheel.

> **CUDA 12 is not supported.** SGLang 0.5.18 is a CUDA 13 build (it requires
> `cuda-python>=13` and `flashinfer_python[cu13]`) and publishes no CUDA 12
> wheel, so the `cuda` extra targets CUDA 13 only.

FlashAttention is not installed through the `fa` extra in the same command:
it builds from source and needs `torch` to be importable first. Install it as
a second step on top of the environment above:

```bash
uv pip install ninja packaging
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
```

The published `specforge` release on PyPI predates the hardware extras, so
`pip install "specforge[cuda]"` only works once a release that includes them
is published. Until then, install from source.

### AMD ROCm

A full dependency resolve on ROCm pulls a CUDA PyTorch from PyPI over the
container's ROCm build, so do not resolve dependencies on ROCm. Instead,
install SpecForge into an environment that already provides a ROCm PyTorch and
a ROCm SGLang (an official SGLang ROCm release container is the recommended
base), and install the package **without dependencies** so pip does not pull
CUDA wheels over the working ROCm stack:

```bash
# Inside the ROCm SGLang container
git clone https://github.com/sgl-project/SpecForge.git /workspace/SpecForge
cd /workspace/SpecForge
python -m pip install -e . --no-deps
```

If a later step reports a missing lightweight dependency (for example
`accelerate`), install just that package, also with `--no-deps`.

For the complete container setup and an end-to-end walkthrough covering
installation, data preparation, offline colocated training, online
disaggregated training, and its single-supervisor and split external launch
forms on AMD Instinct GPUs, follow the
[AMD ROCm Tutorial](../basic_usage/AMD/amd_rocm.md).

### Ascend NPU

You need an Ascend host with the driver and CANN installed, plus the NPU build
of SGLang 0.5.18, `sgl_kernel_npu` and `hccl` from that stack; none of those
three are on PyPI. The `npu` extra then installs the matching PyTorch pieces:
a CPU `torch==2.13.0` (Ascend convention: CPU torch plus `torch_npu` for the
device backend), `torch_npu==2.13.0rc1`, and the `triton` / `triton_ascend`
pair that the fused loss kernels require.

The CPU torch wheel lives on the PyTorch index, not PyPI. uv reads that
routing from `[tool.uv.sources]` for a source install; pip does not, so pass
the index explicitly. Use Python 3.11 or older: `triton_ascend` publishes no
wheels for 3.12.

::: code-group

```bash [uv]
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
uv venv -p 3.11 --seed
source .venv/bin/activate
uv pip install -e ".[npu]"
```

```bash [pip]
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[npu]" --extra-index-url https://download.pytorch.org/whl/cpu
```

:::

Use `--extra-index-url` rather than `--index-url`: the other pins (`torch_npu`,
`triton_ascend`, the base dependencies) still come from PyPI. Do not add
`--pre`: the `torch_npu` pre-release is an exact pin and resolves without it,
while a global `--pre` pulls pre-release builds of unrelated packages. When
installing the PyPI release with uv, pass the same `--extra-index-url`,
because the published package carries no source routing.

The checked-in
[`qwen3.5-4b-dflash-online-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml)
and
[`qwen3.5-4b-domino-online-npu.yaml`](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/online/disaggregated/external/qwen3.5-4b-domino-online-npu.yaml)
recipes use external SGLang server capture with SDPA consumers. Install a
compatible SGLang/Mooncake service first. The unified launcher detects the NPU
device, self-launches the process count recorded in YAML, and selects HCCL; see
the [training guide](../basic_usage/training.md#cuda-rocm-and-ascend-npu) and
the [Ascend NPU Tutorial](../basic_usage/Ascend/ascend_npu.md).
