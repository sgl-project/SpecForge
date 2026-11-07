# Tuned fused-MoE Triton configs for the Qwen3.8-27B DSpark MoE drafter

SGLang runs an MoE drafter's routed experts through its fused MoE Triton kernel
(`fused_experts`, also what the `FusedMoE` module uses on NVIDIA GPUs with the
Triton runner). The kernel picks its tile parameters (`BLOCK_SIZE_M/N/K`,
`GROUP_SIZE_M`, `num_warps`, `num_stages`) per token count from a JSON named
`E=<experts>,N=<expert width>,device_name=<GPU>.json`. Without one it logs

    Using default MoE kernel config. Performance might be sub-optimal! Config file not found at .../E=512,N=512,device_name=NVIDIA_H200.json

and uses a generic tile shape. SGLang ships no config for the
`RadixArk/qwen38-dspark-moe-*` drafters (512 routed experts of width 512,
top-10, hidden 5120, `Qwen3MoeDSparkModel`), so every server that serves them
has been running untuned. This directory adds the tuned files for H200 and
GB300 plus the recipe to make one for another GPU.

## Files

| file | drafter shape | GPU | tuned with |
|---|---|---|---|
| `configs/triton_3_7_1/E=512,N=512,device_name=NVIDIA_H200.json` | 512 x 512, top-10 (Qwen3.8-27B DSpark MoE) | H200 | sglang 0.5.20, triton 3.7.1, batch sizes 7 8 14 16 28 32 56 64 112 128 224 256 448 512, 75 min on 5 GPUs |
| `configs/triton_3_7_1/E=512,N=512,device_name=NVIDIA_GB300.json` | same | GB300 | sglang 0.5.20, triton 3.7.1, batch sizes 7 14 28 56 112 224 448, 33 min on 4 GPUs |
| `tuning/qwen38-dspark-moe/config.json` | the shape as a `Qwen3MoeForCausalLM` config for the tuner | | |
| `tuning/tune_fused_moe_config.sh` | wrapper around SGLang's `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` | | |

The configs only depend on the expert shape (E, N), the GPU and the Triton
version, not on the checkpoint, so they apply to every drafter with 512
experts of width 512 (all `qwen3.8-27b-dspark-moe*` configs). Files are bound
to the GPU model: a B200 needs its own run of the tuner.

## Using them

Either copy the files into SGLang's config directory (keeps SGLang's own
configs for an MoE target):

```bash
C=$(python -c "import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as m, os; print(os.path.dirname(m.__file__))")/configs/triton_3_7_1
cp patches/sglang/moe_configs/configs/triton_3_7_1/E=512,N=512,device_name=*.json "$C/"
```

or point SGLang at this directory (it replaces SGLang's own config directory,
so only do this when the target has no MoE of its own):

```bash
SGLANG_MOE_CONFIG_DIR=$PWD/patches/sglang/moe_configs python -m sglang.launch_server ...
```

The server log then shows `Using MoE kernel config from .../E=512,N=512,device_name=NVIDIA_H200.json`
instead of the "default MoE kernel config" warning. Nothing else changes:
the config only selects tile sizes, so routing, weights and accept length are
identical to the untuned server.

## Measured effect

Kan's 3-epoch MoE drafter (`RadixArk/qwen38-dspark-moe-3ep-cont-step9916`,
served through SGLang's `TopK` + `FusedMoE` module path) against the official
dense drafter (`RadixArk/Qwen3.8-27B-DSpark`), same target, same caps (mamba
cache 160, max 32 running requests, triton attention, DSPARK block 7), untuned
and tuned servers started together and benchmarked in one round
(`specforge benchmark`, gsm8k / mt-bench, 100-200 prompts, 512 new tokens).
Target `Qwen/Qwen3.8-27B-FP8` for both rows below; accept lengths are
identical between untuned and tuned.

| GPU | dataset | c | dense v1 tok/s | MoE untuned | MoE tuned | tuning gain |
|---|---|---|---|---|---|---|
| H200 | gsm8k | 1 / 8 / 32 | 330 / 1609 / 3069 | 318 / 1398 / 2711 | 320 / 1413 / 2738 | +0.6% / +1.1% / +1.0% |
| H200 | mt-bench | 1 / 8 / 32 | 209 / 1119 / 2075 | 204 / 970 / 1842 | 205 / 979 / 1859 | +0.5% / +0.9% / +0.9% |
| GB300 | gsm8k | 1 / 8 / 32 | 427 / 1823 / 4423 | 407 / 1616 / 3823 | 407 / 1626 / 3840 | +0.1% / +0.6% / +0.5% |
| GB300 | mt-bench | 1 / 8 / 32 | 273 / 1373 / 3060 | 262 / 1200 / 2628 | 263 / 1224 / 2669 | +0.2% / +1.9% / +1.6% |

Per draft layer the tuned config is 10% faster at 7 tokens and 3-5% at 56-448
tokens (H200 microbenchmark), but the draft MoE FFN is only 9-21% of a
speculative step, so end to end it is worth 0.5-2%. The remaining gap to the
dense drafter (-4% at c=1, -11% to -14% at c=8/32) is the expert-weight
traffic of the 512 x 512 shape (a 7-token block touches ~65 experts, 1 GB per
layer, vs 0.5 GB for the dense MLP), which no tile config changes.

## Reproducing a config

```bash
bash patches/sglang/moe_configs/tuning/tune_fused_moe_config.sh /path/to/sglang   # defaults: qwen38-dspark-moe shape, batch sizes 7..448
```

The script runs the tuner from its own directory (it imports a sibling
`common_utils.py`), with `--tp-size 1 --dtype auto` and the DSpark token
counts per step as batch sizes; it uses every visible GPU. For fp8 experts add
`--dtype fp8_w8a8 --per-channel-quant` (the file name then carries
`dtype=fp8_w8a8,per_channel_quant=True`). The flashinfer TRT-LLM MoE runner
(`--moe-runner-backend flashinfer_trtllm`, SGLang's default for bf16 MoE on
Blackwell) does not read these files.
