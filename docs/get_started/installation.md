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
PyTorch and a ROCm SGLang, and install the package **without dependencies** so
pip does not pull CUDA wheels over the working ROCm stack.

The recommended base is an official SGLang ROCm release container. These ship a
ROCm PyTorch and an editable ROCm SGLang build, so SpecForge only needs to be
cloned and installed on top.

#### Step 1: Pull the image for your accelerator

The accelerator is baked into the tag, so use the image that matches your
hardware:

```bash
# AMD Instinct MI300X (gfx942)
docker pull lmsysorg/sglang:v0.5.14-rocm720-mi30x

# AMD Instinct MI355X (gfx950)
docker pull lmsysorg/sglang:v0.5.14-rocm700-mi35x
```

#### Step 2: Start the container

Expose the ROCm device nodes (swap in the tag for your accelerator). Use `--name`
and omit `--rm` so the checkout survives across sessions:

```bash
docker run -it --name specforge \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16g \
  lmsysorg/sglang:v0.5.14-rocm720-mi30x \
  bash
```

Re-enter the running container later with `docker exec -it specforge bash`.

#### Step 3: Clone and install SpecForge

Inside the container, clone SpecForge into `/workspace/SpecForge` and register it
in editable mode without touching the image's torch/sglang:

```bash
git clone https://github.com/sgl-project/SpecForge.git /workspace/SpecForge
cd /workspace/SpecForge
python -m pip install -e . --no-deps
```

#### Step 4: Apply the capture patch (online runs only)

These images pin SGLang to exactly `0.5.14` (editable at `/sgl-workspace/sglang`),
so the online capture patch applies with a plain `git apply`. Skip this step for
offline training, which reads features from disk and needs no capture service:

```bash
cd /sgl-workspace/sglang
git apply /workspace/SpecForge/patches/sglang/v0.5.14/spec-capture.patch
```

#### Step 5: Run training

Use the `sdpa` or `flex_attention` attention backends on ROCm. The `fa`
(flash-attn) and `usp` backends, and `yunchang`-based Ulysses/Ring sequence
parallel (`sp_ulysses_size` / `sp_ring_size` > 1), depend on a CUDA flash-attn
build; the single-GPU / data-parallel path never loads `yunchang`, and selecting
those backends raises a clear error. The checked-in
[`qwen3-8b-eagle3-offline.yaml`](../../examples/configs/qwen3-8b-eagle3-offline.yaml)
recipe already uses `flex_attention`, so it runs on ROCm unchanged as a
single-GPU offline EAGLE3 example; launch it with
`specforge train --config examples/configs/qwen3-8b-eagle3-offline.yaml`.

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
