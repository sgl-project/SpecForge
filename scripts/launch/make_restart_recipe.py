"""Generate a weights-only RESTART recipe for the Qwen3.8 MoE cont2ep run from one of its checkpoints.

Usage: python scripts/launch/make_restart_recipe.py <checkpoint_step> [suffix]
Writes examples/configs/online/disaggregated/managed-local/
  qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step<step><suffix>.yaml
- model.draft_checkpoint_path -> the cont2ep run's <step> checkpoint (weights only: optimizer moments, lr warmup
  and the data position start fresh; managed_local has no training.resume_from)
- num_epochs 1, max_steps = 9916 - step (9916 = 2 x 4958 = the two continuation epochs at global batch 256)
- prompt_seed 44 (a fresh corpus shuffle; the dead run's epoch 2 used prompt_seed+1 = 43)
- max_checkpoints 1 by default (272 GB each); raise it in the generated file if the volume has room.
Everything else is identical to qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-from-1ep-v3.yaml.
Used on 2026-09-12 (…-restart-step5500, died on a full /personal) and 2026-09-13 (…-restart-step5500-b, completed).
"""
import sys, yaml
step = int(sys.argv[1])
base = "/personal/SpecForge-qwen38-moe"
src = f"{base}/examples/configs/online/disaggregated/managed-local/qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-from-1ep-v3.yaml"
c = yaml.safe_load(open(src))
prev = c["run_id"]
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
run = f"qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step{step}{suffix}"
out = f"{base}/outputs/{run}"
c["model"]["draft_checkpoint_path"] = f"{base}/outputs/{prev}/{prev}-step{step}"
c["training"].update(num_epochs=1, max_steps=9916 - step, prompt_seed=44, max_checkpoints=1)
c["tracking"]["wandb_name"] = run + "-b200-dp4"; c["tracking"]["wandb_dir"] = out + "/wandb"
c["run_id"] = run; c["output_dir"] = out
c["deployment"]["disaggregated"]["control_dir"] = out + "/control"
c["deployment"]["disaggregated"]["consumer_state_dir"] = out + "/consumer-state"
hdr = (f"# RESTART of {prev} from its step-{step} checkpoint (weights-only warm start) for the remaining\n"
       f"# {9916 - step} steps at global batch 256. prompt_seed 43 gives a fresh corpus shuffle for the remainder;\n"
       "# Mooncake segments 4 x 64 GiB (was 100) with matching watermarks to leave NUMA node 0 headroom for the\n"
       "# trainer ranks' pinned-memory spill. Optimizer moments and lr warmup restart (managed_local cannot resume).\n")
dst = f"{base}/examples/configs/online/disaggregated/managed-local/{run}.yaml"
open(dst, "w").write(hdr + yaml.safe_dump(c, sort_keys=False, width=120)); print("->", dst)
