"""Apply experiment-only logging patches to a MAIN-branch SpecForge tree.

1. train_eagle3.py: print `CURVE step=<s> loss=<weighted ploss>` per micro-step
   (rank0) inside run_backward_and_update. Print-only, no math change.
2. train_dflash.py: print the same CURVE line per step and honor EXP_MAX_STEPS
   (break out of the epoch loop once reached) — main has no --max-num-steps.

Usage: python patch_main.py <tree_root>
"""

import sys

root = sys.argv[1]

# --- train_eagle3.py ---------------------------------------------------------
p = f"{root}/scripts/train_eagle3.py"
src = open(p).read()
anchor = "    ploss.backward()"
patch = (
    "    if not dist.is_initialized() or dist.get_rank() == 0:\n"
    "        print(\n"
    '            f"CURVE step={global_step} "\n'
    '            f"loss={float(ploss.detach().item()) * args.draft_accumulation_steps:.6f}",\n'
    "            flush=True,\n"
    "        )\n"
    "    ploss.backward()"
)
assert anchor in src, "eagle3 anchor not found"
assert "CURVE step=" not in src, "eagle3 already patched"
open(p, "w").write(src.replace(anchor, patch, 1))
print("patched train_eagle3.py")

# --- train_dflash.py ---------------------------------------------------------
p = f"{root}/scripts/train_dflash.py"
src = open(p).read()
anchor = "            (loss / args.accumulation_steps).backward()"
patch = (
    "            if dist.get_rank() == 0:\n"
    '                print(f"CURVE step={global_step} loss={loss.item():.6f}", flush=True)\n'
    "            (loss / args.accumulation_steps).backward()"
)
assert anchor in src, "dflash loss anchor not found"
assert "CURVE step=" not in src, "dflash already patched"
src = src.replace(anchor, patch, 1)

anchor3 = "    total_steps = args.num_epochs * steps_per_epoch"
patch3 = (
    "    total_steps = args.num_epochs * steps_per_epoch\n"
    '    total_steps = int(os.environ.get("EXP_TOTAL_STEPS", "0")) or total_steps'
)
assert anchor3 in src, "dflash total_steps anchor not found"
src = src.replace(anchor3, patch3, 1)

anchor2 = "            global_step += 1"
patch2 = (
    "            global_step += 1\n"
    '            _exp_max = int(os.environ.get("EXP_MAX_STEPS", "0"))\n'
    "            if _exp_max and global_step > _exp_max:\n"
    '                print(f"[exp] stopping at EXP_MAX_STEPS={_exp_max}", flush=True)\n'
    "                return\n"
)
assert anchor2 in src, "dflash step anchor not found"
src = src.replace(anchor2, patch2, 1)
open(p, "w").write(src)
print("patched train_dflash.py")
