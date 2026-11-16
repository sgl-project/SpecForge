# Third-party notices

eagle-harness contains a few files derived from open-source projects. Each such file names its source and the changes
in a header at its top. This file lists them, reproduces the notices those projects ask derived works to keep, and
points to the full license texts in [`LICENSES/`](LICENSES/).

## Derived files

| eagle-harness file | derived from | project | license |
|---|---|---|---|
| `eagle_harness/kernels/gdn_tree.py`, `eagle_harness/kernels/gdn_tree_conv.py` | `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` (tree links and tables, tree conv1d, conv window and state layout) | sglang, which adapted it from vLLM and causal-conv1d | Apache-2.0 (sglang, vLLM), BSD-3-Clause (causal-conv1d) |
| `eagle_harness/kernels/gdn_tree_recurrence.py` | `python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` (gated delta rule recurrence) | sglang, whose `fla` kernels are adapted from flash-linear-attention | Apache-2.0 (sglang), MIT (flash-linear-attention) |
| `eagle_harness/kernels/gdn_tree_gluon.py` | the recurrence of `gdn_tree_recurrence.py` above, rewritten in Gluon | as above | as above |
| `eagle_harness/kernels/mamba2_tree.py` | `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` (per-token selective state update) and `python/sglang/kernels/ops/mamba/triton_ops/mamba_ssm.py` (parent-state reload of the tree-mode update) | vLLM and sglang, which adapted it from mamba (state-spaces/mamba) | Apache-2.0 (vLLM, sglang, mamba) |
| `eagle_harness/sglang/models/llama_eagle4.py` | `python/sglang/srt/models/llama_eagle3.py` (decoder layer with the concatenated embedding and hidden-state input) | sglang, which adapted it from SafeAILab EAGLE `eagle/model/cnets.py` | Apache-2.0 (sglang, EAGLE) |
| `eagle_harness/sglang/tail_mask.py` | `include/flashinfer/attention/variants.cuh` (`DefaultAttention`, as the attention variant's template) | FlashInfer | Apache-2.0 |
| `eagle_harness/tokenspeed/tree_round/ops/csrc/tvm_ffi_utils.h` | `tvm_ffi_utils.h` (tensor validation and CUDA stream interop) | FlashInfer | Apache-2.0 |

The engine adapters (`eagle_harness/sglang/`, `eagle_harness/vllm/`) also import, subclass and call sglang, vLLM and
FlashInfer classes at run time. That code is not copied into eagle-harness: it comes from the engine you install,
under that engine's license.

## Notices

### sglang

Copyright 2023-2024 SGLang Team. Licensed under the Apache License, Version 2.0
([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Source: <https://github.com/sgl-project/sglang>.

### vLLM

Copyright contributors to the vLLM project. Licensed under the Apache License, Version 2.0
([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Source: <https://github.com/vllm-project/vllm>.

### causal-conv1d

Copyright (c) 2024, Tri Dao. Licensed under the BSD 3-Clause License
([`LICENSES/BSD-3-Clause-causal-conv1d.txt`](LICENSES/BSD-3-Clause-causal-conv1d.txt)).
Source: <https://github.com/Dao-AILab/causal-conv1d>.

### mamba

Copyright (c) 2023, Tri Dao, Albert Gu. Licensed under the Apache License, Version 2.0
([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Source: <https://github.com/state-spaces/mamba>.

### flash-linear-attention

Licensed under the MIT License, whose copyright line and permission notice are in
[`LICENSES/MIT-flash-linear-attention.txt`](LICENSES/MIT-flash-linear-attention.txt).
Source: <https://github.com/fla-org/flash-linear-attention>.

### FlashInfer

The NOTICE file of FlashInfer begins:

```text
FlashInfer
Copyright 2025-2026 NVIDIA
Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
```

Licensed under the Apache License, Version 2.0 ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)).
Source: <https://github.com/flashinfer-ai/flashinfer>.

### EAGLE (SafeAILab)

Copyright 2025 SafeAI Lab (SAIL). Licensed under the Apache License, Version 2.0
([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)). Source: <https://github.com/SafeAILab/EAGLE>.

## Prebuilt native library

The binary wheel ships `eagle_harness/kernels/_native/cu<major>/`, built from `eagle_harness/kernels/csrc/candidate.cu`
with the CUDA runtime linked statically (`-cudart static`, in `eagle_harness/kernels/build.py`). The CUDA runtime is
NVIDIA software under the NVIDIA CUDA Toolkit End User License Agreement (<https://docs.nvidia.com/cuda/eula/>).

That agreement asks for no notice or attribution text in return, so none appears above. Attachment A lists
`libcudart_static.a` among the redistributable portions of the SDK, and the one notice the agreement prescribes
(section 1.1.2) covers modifications and derivative works of its sample source code, which this project does not
ship. The conditions that do apply are met: the library adds material functionality of its own beyond the runtime,
nothing but eagle-harness reaches the runtime it links, the runtime is redistributed unmodified, and no NVIDIA
copyright or proprietary notice has been removed.

Section 1.2 also forbids using the SDK so as to make it subject to a license that requires it to be disclosed in
source form, licensed for the purpose of making derivative works, or redistributable at no charge. The Eagle 4
Source-Available License is not such a license: it is not an open-source license, and its source condition (section
4c) excludes "separable proprietary components that are not derived from or modifications of the Software". No source
distribution carries the runtime in any case, because `MANIFEST.in` prunes `eagle_harness/kernels/_native` from the
sdist and a source install compiles `candidate.cu` against the CUDA toolkit already on the machine.

## Redistributed ROCm wheels

This section is separate from the eagle-harness notices above. The EAGLE-4 ROCm wheel index also serves three
third-party wheels that no vendor index carries in the tested form. They are separate downloads, not part of
eagle-harness, and each keeps the license files of its own source. Every other ROCm dependency is installed from
PyPI or AMD's indexes, not from the EAGLE-4 index.

| Wheel | Project and license | Change |
|---|---|---|
| `torch-2.9.1+rocm7.2.0.lw.git7e1940d4-1eagle-cp310-cp310-linux_x86_64.whl` | PyTorch, as built by AMD for ROCm 7.2. BSD-3-Clause. Its `LICENSE` and `NOTICE` ship unchanged in the wheel. | Modified metadata only: the requirement `triton==3.5.1+rocm7.2.0.gita272dfa8` is replaced by `triton==3.7.0+amd.rocm7.2.0.git89002410`, `WHEEL` gains build tag `1eagle`, and `RECORD` is regenerated (`tools/repack_rocm_torch.py`). Every other file is AMD's original. |
| `sglang_kernel-0.4.7-1eagle-cp310-cp310-linux_x86_64.whl` | SGLang kernel package (`python/sglang/kernels/aot`). Apache-2.0. Copyright 2023-2024 SGLang Team. | Built from unmodified SGLang source for HIP `gfx950`. SGLang's `THIRDPARTYNOTICES.txt` from the same source is added to the wheel's license files (`tools/add_wheel_notices.py`). |
| `amd_aiter-0.1.21.dev48+g4ad998328.d20260918-cp310-cp310-linux_x86_64.whl` | AITER. MIT. Copyright Advanced Micro Devices, Inc. It bundles composable_kernel (MIT, Advanced Micro Devices, Inc.), whose `LICENSE` ships in the wheel. | Built from ROCm/aiter revision `4ad99832` with the patches the SGLang v0.5.20 ROCm image applies: ROCm/aiter commits `7b481fb` and `24a62b1` (tuned GEMM configurations), a `functools.cache` edit and a `torch.Stream` compatibility edit. |

The Torch wheel also contains two libraries that its `LICENSE` file does not name. Both are AMD's builds,
redistributed unchanged: `libaotriton_v2.so` (AOTriton, MIT, Advanced Micro Devices, Inc.,
<https://github.com/ROCm/aotriton>) and `libmagma.so` (MAGMA, BSD-3-Clause, <https://icl.utk.edu/magma/>).
