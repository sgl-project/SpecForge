# Qwen3/Qwen3.6 Domino disaggregated single-sample overfit

This gate checks the smallest end-to-end Domino disaggregated training path:
one patched SGLang capture server regenerates features for one ShareGPT row,
and a separate FSDP trainer repeatedly consumes that row. A companion
post-training gate exports the resulting checkpoint and loads it in real
SGLang DFLASH speculative serving. The historical 8B-named entry points are
shared implementations; `MODEL_PROFILE` selects the defaults.

| Profile | Dataset contract | Chat | Real serving default |
| --- | --- | --- | --- |
| `qwen3-8b` | non-reasoning, t=0 | `qwen`, thinking off | TP=1, GPU 0 |
| `qwen3.6-27b` | reasoning, t=0 | `qwen3.5`, thinking on | TP=1, GPU 0 |

The profile supplies defaults only. Model/data paths and topology knobs
remain overridable. Mask token, capture layers and block size are read from the
selected draft config so the launch command does not duplicate model metadata.

By default, the Qwen3-8B regeneration recipe writes its non-reasoning,
temperature-zero ShareGPT output to:

```text
cache/dataset/sharegpt_train_regen_qwen3_8b_temperature0_non_reasoning.jsonl
```

To regenerate this source first, use the non-reasoning Qwen3-8B recipe:

```bash
bash examples/data_regeneration/run_qwen3_8b_sharegpt_non_reasoning.sh
```

That recipe reports success, error, and skipped rows, requires complete-row
accounting, and strictly validates successful output. Qwen3.6 reasoning
regeneration uses the same implementation:

```bash
MODEL_PROFILE=qwen3.6-27b \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

The overfit wrapper independently enforces the selected profile's reasoning
contract and real post-tokenization loss-mask threshold before starting any
server or trainer process.

Run from the repository root on an eight-GPU host with Mooncake and a patched
SGLang capture server installed:

```bash
SOURCE_DATA_PATH=/path/to/qwen3_8b_non_reasoning_regen.jsonl \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
SERVER_GPUS=0 \
CONSUMER_GPUS=1,2,3,4,5,6,7 \
TRAIN_DP=7 \
bash examples/disagg/run_qwen3_8b_domino_disagg_overfit.sh
```

For the validated Qwen3.6-27B reasoning dataset and full-attention Domino config:

```bash
MODEL_PROFILE=qwen3.6-27b \
SERVER_GPUS=0 SERVER_TP=1 \
CONSUMER_GPUS=1,2,3,4,5,6,7 TRAIN_DP=7 \
bash examples/disagg/run_qwen3_8b_domino_disagg_overfit.sh
```

The launcher intentionally refuses an existing `WORK_DIR`; choose a new path
for every run. It selects one clean single-turn row into `single_sample.jsonl`
and runs the same Qwen tokenizer and SpecForge preprocessing used by the
producer. Selection succeeds only when the resulting loss mask has at least 32
tokens, matching the producer's `2 * block_size` filter, and the rendered row
fits without truncation. Both selection and training use
`--train-only-last-turn`. Selection also writes
`prompt_artifact.json`, which records the exact rendered prompt/target suffix.
The serving client uses that artifact instead of reconstructing a target from
visible content; this preserves Qwen3.6 reasoning while stripping historical
assistant reasoning from the request.

`DISAGG_MAX_PROMPTS=1` limits the producer input to that one source row.
`NUM_EPOCHS` is then set to `DISAGG_MAX_STEPS * TRAIN_DP`, so replay produces
exactly enough sample references for every DP rank at every bounded optimizer
step. The default gate stops at `DISAGG_MAX_STEPS=400`.

Useful bounded overrides are:

```bash
WORK_DIR=outputs/qwen3-8b-overfit-$(date +%Y%m%dT%H%M%S) \
DISAGG_MAX_STEPS=600 \
bash examples/disagg/run_qwen3_8b_domino_disagg_overfit.sh
```

The command exits successfully only when all three conditions hold:

- the final logged loss is at most `0.0001` (displayed as approximately
  `0.0000`);
- final token accuracy is exactly `1.0000` by default;
- a `training_state.pt` checkpoint exists under the fresh consumer output.

The final JSON emitted by `scripts/check_domino_overfit.py` records the step,
loss, token accuracy, and checkpoint path. A nonzero exit means this gate did
not pass even if the trainer process itself exited cleanly.

## Real DFLASH serving gate

Run the strict serving check against the fresh overfit checkpoint and its
`prompt_artifact.json`:

```bash
MODEL_PROFILE=qwen3-8b \
CHECKPOINT_PATH=/path/to/consumer/<run-id>-step400 \
PROMPT_ARTIFACT_PATH=/path/to/run/prompt_artifact.json \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
SGLANG_ROOT=/path/to/domino-capable/sglang \
SGLANG_PYTHON=/path/to/sglang-env/bin/python \
bash examples/disagg/run_qwen3_8b_domino_real_serving_gate.sh
```

Alternatively, set `RUN_REAL_SERVING_GATE=1` on the training wrapper and set
`SERVING_SGLANG_ROOT` / `SERVING_PYTHON` when serving uses a separate SGLang
environment. The wrapper releases the capture server and trainer GPUs before
starting the real serving process.

The runner exports the runtime checkpoint with the target embedding, changes
the exported architecture to SGLang's `DFlashDraftModel`, and launches a real
server with `--speculative-algorithm DFLASH` and block size 16. Its client only
uses `/v1/chat/completions`, sends `enable_thinking=false`, and removes
`reasoning_content` from request history. For `qwen3.6-27b`, the same client
sends `enable_thinking=true`, combines response `reasoning_content + content`,
and compares it with the artifact's rendered target suffix. The launcher also
uses `--reasoning-parser qwen3`, TP=1, and a 2048-token context by default for
that profile.

This is a strict per-request gate. The SGLang checkout must support
`return_meta_info=true` on the OpenAI chat API and expose
`choices[0].meta_info.spec_accept_length`; aggregate `/server_info` acceptance
statistics are not accepted as a substitute. The result passes only when:

- `/server_info.speculative_algorithm` is `DFLASH`;
- the request contains no `reasoning_content`;
- the choice reports `spec_accept_length >= 16`;
- the generated output matches at least the first 16 target tokens.

The auditable result is written to `serving_gate.json`. An older SGLang checkout
that omits choice metadata fails explicitly instead of being reported as real
per-request serving success.
