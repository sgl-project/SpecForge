"""LongSpark: draft-KV-free parallel drafting over captured target K/V."""

FEATURES = (
    "input_ids",
    "loss_mask",
    "hidden_states",
    "target_last_hidden_states",
    "target_keys",
    "target_values",
)

# Draft block length. The dataset filter runs before the draft config is
# loaded, so it uses this value; the model provider checks the config matches.
BLOCK_SIZE = 7
