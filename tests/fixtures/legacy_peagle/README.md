# Trained legacy P-EAGLE fixture

Synthetic CPU FP32 checkpoint generated with `generate.py` on PR #1109
(commit `a5ec15a0`). One SGD step updates token row 1 while padding row 0 stays
unchanged. The reference contains token IDs, target features and post-step outputs.
Run the generator with that checkout on PYTHONPATH and an output directory argument.
