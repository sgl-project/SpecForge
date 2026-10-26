# Data Preparation

## Overview

Data is an important aspect of speculative decoding as the quality of the dataset directly affects the acceptance rate of the draft model. In this section, we will introduce how to prepare the dataset for both online and offline training.

## Canonical Dataset Presets

`scripts/prepare_data.py --dataset NAME` exposes the same 19 text presets
used by the checked-in recipes:

| Family | Presets |
| --- | --- |
| General chat | `ultrachat`, `sharegpt`, `eaglechat`, `perfectblend`, `perfectblend-llama3.1-8b-instruct`, `perfectblend-llama3.3-70b-instruct`, `perfectblend-llama4-scout-instruct`, `perfectblend-llama4-maverick-instruct`, `magpie-qwen2.5-pro-1m-v0.1`, `nebius-llama31-8b-infinity-instruct` |
| Reasoning, math, and code | `opc`, `gsm8k`, `hendrycks_math`, `math_qa`, `codealpaca-20k`, `opencodeinstruct`, `magicoder-evol-instruct`, `sciq`, `camel` |

VLM data preparation and training, including ALLaVA and ShareGPT4V, are not
supported.

Every successful preset writes the same stable `id` + `conversations` JSONL
contract. Use `--split-eval` to create a deterministic 95/5 train/eval split;
`opc` additionally accepts `--opc-subset`.

Run the script from the repository root:

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

By default, each command writes `<preset>_train.jsonl` under `cache/dataset`.
Use `--output-path` to choose another output directory and `--sample-size` to
cap the number of source rows.

For a local ShareGPT-format dataset, pass a JSON or JSONL file with
`--data-path`:

```bash
python scripts/prepare_data.py \
    --dataset sharegpt \
    --data-path ./raw_sharegpt.jsonl \
    --output-path ./cache/dataset
```

Each raw row must contain an `id` and a `conversations` list whose messages use
ShareGPT's `from` and `value` keys. Custom paths are intentionally limited to
the `sharegpt` preset.


## ↩️ Regenerate Datasets

When training speculative decoding draft models for a specific target model, instead of using the original dataset, we can regenerate the assistant responses using the target model to better align the draft model with the target model's output distribution. This will improve the acceptance rate of the draft model and the overall performance of the speculative decoding. According to the [EAGLE1 paper](https://arxiv.org/pdf/2401.15077), the EAGLE method is not very sensitive to the dataset quality, which means the performance is still good even if you use the original dataset. However, if you are looking for optimal performance in the production environment, it is recommended to regenerate the dataset using the target model.

The regeneration utility uses the OpenAI-compatible client to call SGLang's
HTTP API. Install it in the source-checkout environment before running the
repository script:

```bash
pip install -e '.[data]'
```

We can follow the following steps to regenerate the dataset. In the example below, we will use `meta-llama/Llama-3.1-8B-Instruct` as an example, you can replace it with your own target model.

1. Start the SGLang server for the target model.

```shell
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --cuda-graph-max-bs 128 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --port 30000
```

2. Regenerate the dataset using the `regenerate_train_data.py` script.

```shell
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 98304 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
```

For reasoning models, add `--reasoning save` to store `reasoning_content` in the regenerated dataset. To use a reasoning model with thinking disabled, add `--reasoning disable`, which forwards `chat_template_kwargs.enable_thinking=false` to the SGLang server and does not save `reasoning_content`.

### Multi-turn structured reasoning

A regenerated multi-turn reasoning conversation stores several assistant
targets in one row. When training uses last-turn-only loss masking, only the
final assistant target is supervised. Convert the validated conversation-level
output into one generation-event row per assistant turn so every reasoning
target is trained with the visible history available at its serving boundary:

```bash
python scripts/expand_reasoning_conversations.py \
    --input-file-path ./cache/dataset/sharegpt_train_regen_reasoning.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen_reasoning_exploded.jsonl
```

Each event ends at its current assistant target and preserves that turn's
`reasoning_content` and visible `content`. Historical assistant messages keep
only visible `content`; their hidden reasoning is removed from the event
context. Train the exploded output with `data.train_only_last_turn: true` in
the typed run config.

The converter accepts only successful rows with non-empty IDs, message content,
and assistant `reasoning_content`. Invalid input is written to a skipped JSONL;
if any turn is invalid, the entire source conversation is skipped. Output files
must be fresh and distinct from the input.

For maximum performance, we recommend to scale the number of GPUs to regenerate the dataset in data parallel mode. To do this, you can simply add more server addresses to the `--server-address` argument, e.g. `--server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003`.

### Qwen ShareGPT recipes

The Qwen recipe wraps regeneration, complete-row accounting, and output
validation. It expects one or more SGLang servers to already be running; it does
not launch or stop the servers itself.

For Qwen3-8B non-reasoning regeneration, start a target server in one terminal:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run the recipe from
the SpecForge repository root:

```bash
MODEL_PROFILE=qwen3-8b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
OUTPUT_FILE=./cache/dataset/sharegpt_train_regen_qwen3_8b_temperature0_non_reasoning.jsonl \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

This profile sets temperature to zero, disables thinking through
`chat_template_kwargs.enable_thinking=false`, and rejects non-empty structured
reasoning in successful rows.

For Qwen3.6-27B structured-reasoning regeneration, start SGLang with the Qwen3
reasoning parser so the OpenAI-compatible response includes
`reasoning_content`:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run:

```bash
MODEL_PROFILE=qwen3.6-27b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
OUTPUT_FILE=./cache/dataset/sharegpt_train_regen_qwen3.6-27b_temperature0_reasoning.jsonl \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

The default maximum completion lengths are 4096 tokens for Qwen3-8B and 32768
tokens for Qwen3.6-27B. The server context length must accommodate both the
rendered prompt and `MAX_TOKENS`; override `MAX_TOKENS` when using a shorter
server context.

`INPUT_FILE` may point to another dataset without changing the recipe as long
as it is JSONL in the conversation format documented below, with a non-empty
string `id` and alternating `user`/`assistant` messages. Always choose a fresh
`OUTPUT_FILE`: the recipe deliberately refuses to overwrite or resume an
existing run.

For an output such as `dataset_regen.jsonl`, the recipe produces:

- `dataset_regen.jsonl`: successful regenerated training rows;
- `dataset_regen_error.jsonl`: request or generation errors;
- `dataset_regen_skipped.jsonl`: schema-invalid inputs and outputs excluded by the
  selected reasoning contract.

The run passes accounting only when
`success + error + skipped == input`. It prints the success fraction without a
fixed minimum threshold, then strictly validates every successful row,
including its conversation structure, reasoning contract, and absence of raw
`<think>` markers.

To distribute regeneration across independently launched servers, provide all
addresses as a space-separated list, for example:

```bash
SERVER_ADDRESSES="localhost:30000 localhost:30010" \
CONCURRENCY=64 \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

## Prepare your own dataset

Besides the provided datasets, you can also prepare your own dataset. We support two formats:

### Option 1: Conversation Format

You should prepare the dataset in jsonl format and the schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

### Option 2: Pre-formatted Text Format

If you already have conversations formatted with a specific chat template, you can use the pre-formatted text directly:

```json
{
    "id": "xxxx",
    "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"
}
```

This format is useful when you have pre-formatted prompts that were used during training of the target model and have raw generations from the target model.

To use preformatted datasets, set `data.is_preformatted: true` in the run
config. `data.chat_template` is still required and must match the template used
to create the text. SpecForge uses it to identify assistant spans and build the
loss mask.

```bash
# After copying an online disaggregated example YAML, set these data fields:
# data:
#   train_data_path: ./your_preformatted_dataset.jsonl
#   is_preformatted: true
#   chat_template: llama3
specforge train --config ./my-eagle3-disaggregated.yaml
```

## Prepare offline target features

Offline EAGLE3, DFlash, DFlash2, Domino, and DSpark runs consume
target features created by `scripts/prepare_hidden_states.py`. Select the same
strategy and draft
configuration that the later training recipe uses; these values determine both
the target layers captured and the checkpoint schema. For example, prepare the
checked-in Qwen3-8B DFlash recipe with:

```bash
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --strategy dflash \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config configs/qwen3-8b-dflash.json \
    --data-path ./cache/dataset/sharegpt_train.jsonl \
    --output-path ./cache/hidden_states/qwen3-8b-dflash-sharegpt \
    --chat-template qwen \
    --max-length 3072 \
    --tp-size 1 \
    --batch-size 32
```

Use these strategy/model pairs for the checked-in offline colocated recipes:

| Strategy | Target model | Draft config | Output path used by the recipe |
| --- | --- | --- | --- |
| EAGLE3 | `Qwen/Qwen3-8B` | `configs/qwen3-8b-eagle3.json` | `cache/hidden_states/qwen3-8b-sharegpt` |
| DFlash | `Qwen/Qwen3-8B` | `configs/qwen3-8b-dflash.json` | `cache/hidden_states/qwen3-8b-dflash-sharegpt` |
| Domino | `Qwen/Qwen3-8B` | `configs/qwen3-8b-domino.json` | `cache/hidden_states/qwen3-8b-domino-sharegpt` |
| DSpark | `Qwen/Qwen3-4B` | `configs/qwen3-4b-dspark.json` | `cache/hidden_states/qwen3-4b-dspark-sharegpt` |

`--strategy` defaults to `eagle3` for compatibility. Pass it explicitly in
reproducible jobs. `--draft-model-config` is required for Domino and DSpark;
passing the same explicit config used by training is recommended for every
strategy. Capture layers come from the resolved draft config. EAGLE3 retains
target-derived defaults for legacy configs that do not define
`eagle_config.eagle_aux_hidden_state_layer_ids`.

Each output record contains the strategy's exact offline feature contract:

| Strategy | Tensors in each `.ckpt` or `.ckpt.gz` record |
| --- | --- |
| EAGLE3 | `input_ids`, `loss_mask`, `hidden_state`, `aux_hidden_state` |
| DFlash, DFlash2, and Domino | `input_ids`, `loss_mask`, `hidden_states` |
| DSpark | `input_ids`, `loss_mask`, `hidden_states`, `target_last_hidden_states` |

For the DFlash family, `hidden_states` concatenates the target layers selected
by the draft config. DSpark additionally stores the target model's final hidden
state for its L1 and confidence objectives. Keep each strategy in a separate
output directory; the offline reader validates the contract instead of
silently adapting incompatible features.

DFlash2 feature preparation still uses `--strategy dflash`; pass a
`DFlash2DraftModel` config such as `configs/qwen3.6-27b-dflash2.json` so capture
uses the intended `target_layer_ids`. Its convolution, selector, and selector
loss schedule are draft/trainer concerns and do not add tensors to the offline
record.

D-PACE uses `training.strategy: dflash` and the DFlash feature schema. DTA also
uses `--strategy dflash`, but feature preparation must receive its DTA draft
config so the captured layer contract matches the training run.

For preformatted input, add `--is-preformatted` to the same command and keep
`--chat-template` aligned with the template already applied to the text:

```bash
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --strategy eagle3 \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config configs/llama3-8B-eagle3.json \
    --data-path ./your_preformatted_dataset.jsonl \
    --output-path ./cache/hidden_states/llama3.1-8b-eagle3 \
    --chat-template llama3 \
    --is-preformatted \
    --max-length 2048
```

The preparation world size only parallelizes capture. The subsequent offline
run still uses the one `specforge train` entry and self-launches the
data-parallel or EAGLE3 USP topology recorded under `deployment.trainer`.
Launch the matching recipe after its `data.hidden_states_path` points at the
generated directory:

```bash
specforge train --config examples/configs/offline/colocated/qwen3-8b-dflash.yaml
```

See the [Training](training.md) guide for the complete run schema and supported
combinations.

### Targets without a local SGLang engine

`--target-backend` selects what runs the target during capture. The default,
`sglang`, is described above. Two more backends exist for targets that a local
SGLang build cannot run yet, currently Qwen3.8-Flash-Next (`qwen4_exp`); both
support the DFlash and DSpark capture methods and keep the rendering, loss
masks, record layout and resumption of the SGLang path.

**`hf`: the Transformers reference model.** One process drives the whole node
(`torchrun --nproc_per_node 1`, `--tp-size 1`): consecutive decoder layers are
spread over the visible accelerators (`--hf-num-devices`), so only one device
computes at a time. Flash-Next's ~95 GiB n-gram embedding stays in host memory,
where its lookup runs; load the BF16 checkpoint with at least ~400 GiB of host
memory. It needs `transformers>=5.16.1` in the capture environment only. Rows
run one at a time over their real tokens and are saved at their real length,
so `--batch-size` only sets how many rows a step reads. Two reference paths are
replaced for speed (a vectorized QSA indexer and one-readback expert
dispatch), and the capture is bit-identical to the reference model running the
same paths.

**`vllm`: a running vLLM / vLLM-Ascend hidden-state server.** Start the server
with the `extract_hidden_states` speculative method and
`ExampleHiddenStatesConnector`, keeping the connector's
`use_synchronization_lock` on (its default): the connector writes each file
asynchronously and its lock is the only sign the file is complete, so the
backend refuses a capture without one. The backend sends each row's real token
ids to `/v1/completions` with `max_tokens=1`, waits on the connector's file
lock and converts the `[T, K, H]` capture into the strategy's record. It loads no model
(`SPECFORGE_DEVICE=cpu` is enough) and only reads the target's `config.json`;
when the installed Transformers does not know the model type, the raw file is
read instead, as training does, so the repository's pinned release is enough.

- The server's layer ids are the draft's `target_layer_ids` plus one, followed
  by `num_hidden_layers`: vLLM numbers a layer's *input*, SpecForge a layer's
  *output*, and the last slot is the state `lm_head` reads. For Flash-Next that
  is `[4, 16, 24, 36, 44, 48]`. The backend prints the expected value and
  refuses a capture with the wrong layer count.
- Echoed and saved token ids, sequence length and finiteness are checked per
  row. Rows are sent unpadded and saved at their real length, so set
  `--batch-size` equal to `--vllm-concurrency`, at most the server's
  `--max-num-seqs`; keep `--max-length` below its `--max-model-len`.
- `--vllm-path-map SERVER_PREFIX=LOCAL_PREFIX` translates capture paths when the
  server writes them in another container; `--vllm-keep-captures` keeps the
  server's files for debugging.

Upstream vLLM and vLLM-Ascend cannot yet serve Qwen3.8-Flash-Next this way:
vLLM's Qwen4Exp config check admits only its native MTP and n-gram proposers,
and the model does not emit auxiliary hidden states. The
[`qwen38-capture`](https://github.com/fg11991/vllm-ascend/tree/qwen38-capture)
branch of `fg11991/vllm-ascend` carries the changes, chiefly
[`8cc7129`](https://github.com/fg11991/vllm-ascend/commit/8cc712978334a47c670aeb82dc3ef0d091c53553)
(allow `extract_hidden_states` for Qwen4Exp) and
[`1de1b0b`](https://github.com/fg11991/vllm-ascend/commit/1de1b0b5ae6e39f13674a08ac4975c197bfb475c)
(serve the auxiliary hidden states), plus KV-cache sizing fixes for the
capture path. It has been run on one Atlas 800 A3 node: W8A8 weights, TP8,
`--enforce-eager`, native MTP off, `--max-model-len 2304`.

```bash
SPECFORGE_DEVICE=cpu torchrun --nproc_per_node=1 \
    scripts/prepare_hidden_states.py \
    --strategy dspark \
    --target-model-path /models/Qwen3.8-Flash-Next \
    --draft-model-config configs/qwen3.8-flash-next-dspark-moe64.json \
    --data-path ./cache/dataset/train.jsonl \
    --output-path ./cache/hidden_states/qwen3.8-flash-next-dspark \
    --chat-template qwen3.5 \
    --max-length 2048 \
    --tp-size 1 \
    --target-backend vllm \
    --vllm-url http://127.0.0.1:8000 \
    --vllm-served-model-name qwen3.8-flash-next \
    --vllm-concurrency 4 \
    --batch-size 4
```

The MoE drafter in `configs/qwen3.8-flash-next-dspark-moe{64,512}.json` mirrors
the target's MTP layer, so it can start from it:
`scripts/warm_start_from_mtp.py` writes a pretrained draft directory (gated
attention, router, routed and shared experts copied; the rest freshly
initialized) for `model.draft_checkpoint_path`. The 64-expert arm names the
MTP experts it starts from with `--expert-ids` or `--expert-selection stride`.


## Handling Multiple Datasets

If you have multiple datasets, you can just merge them into the one jsonl file. For example, you can do something like this

```bash
cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl
```
