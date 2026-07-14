# Training and serving gates

These helpers remain independent of the training implementation. Training now
starts only through `specforge train`; the removed method-specific Python
launchers are not required by these gates.

## One-sample overfit gate

1. Select a clean sample and capture its exact serving prompt boundary:

   ```bash
   python scripts/gates/select_overfit_sample.py \
       --input-data-path "$SOURCE_DATA_PATH" \
       --output-data-path "$WORK_DIR/single_sample.jsonl" \
       --prompt-output-path "$WORK_DIR/prompt.json" \
       --model-path "$TARGET_MODEL_PATH" \
       --chat-template qwen \
       --min-loss-tokens 32 \
       --require-untruncated
   ```

2. Point a unified online-disaggregated YAML config at `single_sample.jsonl`,
   then launch its producer and consumer with
   `examples/disagg/run_online.sh`. Pass dotted overrides such as
   `training.max_steps=400`, `training.log_interval=1`, and
   `training.save_interval=400` after the role argument.

3. Check the final unified trainer log and checkpoint:

   ```bash
   python scripts/gates/check_overfit_metrics.py \
       --log-path "$WORK_DIR/train.log" \
       --checkpoint-root "$WORK_DIR/checkpoints" \
       --expected-step 400 \
       --max-loss 0.0001 \
       --min-accuracy 1.0
   ```

## Real DFLASH serving gate

Export the resulting checkpoint through `specforge export --to hf`, start an
SGLang server with the exported draft and `--speculative-algorithm DFLASH`, then
run:

```bash
python scripts/gates/run_dflash_chat_serving_gate.py \
    --server-url "http://127.0.0.1:30000" \
    --model-path "$TARGET_MODEL_PATH" \
    --served-model "$SERVED_MODEL" \
    --prompt-json-path "$WORK_DIR/prompt.json" \
    --output-path "$WORK_DIR/serving-gate.json" \
    --block-size 16 \
    --max-tokens 16
```

The serving checker validates the actual chat endpoint, DFLASH server identity,
choice-level acceptance metadata, and a target-prefix match. It intentionally
does not start or stop training and serving processes.
