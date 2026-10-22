# Draft module precision

The model's global `dtype` applies to **new parameters only**. Existing weights
retain the dtype of each tensor in the checkpoint, even if `config.json` says
`"dtype": "bfloat16"` and some weight tensors are FP32. The tensor data, rather
than a single config field or safetensors index entry, is authoritative.

Hugging Face loading (including sharded safetensors), warm starts, full training
resume, and copied target embedding/head weights preserve source dtypes. The
training builders move loaded models to the device without globally recasting
them. State loading retains parameter identities and tied aliases; missing
parameters keep their initialization dtype. The optimizer's FP32 master copies
are independent of this policy and copy back to each parameter's own dtype.

## Initialization and RoPE buffers

`AutoDraftModel.from_config` uses the Transformers initialization dtype context.
P-EAGLE uses the same construction path. The requested dtype is the default
for new tensors, while an explicit dtype in a module constructor takes priority.
The builders do not cast the complete model after construction or loading.

DFlash-family drafts use the upstream `Qwen3RotaryEmbedding` directly. Its
frequency buffers are initialized in FP32 independently of BF16 model weights.
No additional precision setting or custom casting wrapper is needed. Device-only
moves preserve these dtypes. Angles and trigonometry follow the upstream FP32
implementation; returned cos/sin tensors follow the activation dtype.

These frequency buffers are non-persistent and are regenerated from the model's
RoPE configuration when loading an export. They are not checkpoint parameters.
A user's explicit `model.to(dtype=...)`, `.half()`, or `.bfloat16()` still follows
normal PyTorch semantics and can cast floating-point parameters and buffers.
SpecForge avoids these whole-model casts in the initialization/loading path.

## New training, old checkpoints, and inference

Start a fresh run with a new run ID, output directory, control directory, consumer
state directory, and feature-store ID after changing numerical behavior. Keep
previous exports and training checkpoints intact. Loading old weights preserves
their tensor dtypes; it cannot undo earlier training with rounded frequencies.
Do not assume an exact numerical continuation of such an old training run merely
because its state can be loaded. The removed precision option is no longer part
of the training resume contract.

SGLang has its own RoPE implementation. Native FP32 frequencies alone do not
establish end-to-end parity: positions, scaling, attention masks, kernels, and
KV precision must also agree. No acceptance-rate improvement is claimed here.

Run `python -m unittest tests.test_modeling.test_draft_precision` to check explicit
module dtypes, initialization-context restoration, native and dynamic RoPE,
export/reload, BF16 forward/backward, and device-only moves on CUDA when available.

## Distributed training and compute precision

FSDP does not apply a global parameter dtype override. Mixed-parameter models
receive homogeneous dtype wrapping boundaries so FP32 weights stay FP32 during
forward/backward and after optimizer updates. Homogeneous models retain their
existing block wrapping. FSDP1 cannot flatten differently typed parameters
owned directly by the same leaf module; this case raises an explicit error
instead of casting weights. Use NO_SHARD/DDP or restructure that module.

Preserving weight storage dtype does not automatically define operator compute
precision. A model that intentionally mixes FP32 and BF16 modules must specify
activation casts at those boundaries. This change does not insert arbitrary
casts into an architecture, modify SGLang's loader, or make every possible
mixed-dtype checkpoint numerically compatible with every kernel. Explicit user
calls to ordinary PyTorch `model.to(dtype=...)` still request a conversion;
SpecForge's loading/training builders no longer perform that conversion after
loading checkpoint tensors.

Run `python -m unittest tests.test_modeling.test_checkpoint_precision` to check
mixed checkpoint loading and two-GPU FSDP/NO_SHARD forward/backward and optimizer
updates (GPU tests skip when fewer than two GPUs are available).

## LM head and logits precision

The frozen target projection retains the checkpoint weight dtype and casts its
input to that dtype for GEMM. For example, BF16 weights produce BF16 base logits;
FP32 weights produce FP32 base logits. The global initialization dtype does not
rewrite the loaded head.

`TargetHead` converts the projection result to FP32, matching SGLang's default
(non-`enable_fp32_lm_head`) logits processing. Casting the result to FP32 is not
an FP32 projection: low-precision GEMM output has already been rounded.

The dense DSpark draft path has a different boundary. It adds the Markov bias
to native-dtype base logits before converting to FP32 for probabilities and
cross-entropy/DPACE. Promoting base logits before that addition would change
BF16 rounding and diverge from the installed SGLang DSpark implementation.
FP64 diagnostic loss inputs retain their precision. This does not reproduce
optional FP32-head flags or specialized DeepSeek-native draft head kernels.

Run `python -m unittest tests.test_modeling.test_lm_head_precision` for projection,
correction-order and backward checks on CPU and CUDA (when available). This
checks arithmetic boundaries, not end-to-end acceptance or TP kernel parity.
