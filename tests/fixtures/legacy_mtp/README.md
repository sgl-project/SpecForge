# MTP migration regression fixture

Synthetic CPU FP32 weights and inputs from seed 731. The reference outputs and
`logits.square().mean()` gradients were regenerated with main's #1014 corrections:
declared theta 10000000, fp32 rotary frequencies, and the corrected attention layout.
The embedding is frozen. This fixture tests the corrected model's migration;
it does not claim exact resume compatibility with runs trained before #1014.

`original_config.json` preserves the retired full-size recipe for codec tests.
