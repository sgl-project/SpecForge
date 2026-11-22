#!/usr/bin/env bash
# Run from the fork. Each phase exits before the next takes the single GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=examples/configs/offline/colocated/qwen3.5-4b-mmflash-a100.yaml
cache=cache/mmflash
data=$cache/chartqa
target=$cache/target
out=outputs/mmflash-smoke
pin=5b3fa3fe28f5bc6d7dfe75934a156c52b865dc8c
author=${MMFLASH_REFERENCE:-$cache/author}
port=${PORT:-30000}
url=http://127.0.0.1:$port
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export SPECFORGE_DEVICE=cuda

preflight() {
    git rev-parse HEAD
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    python - <<'PYENV'
from importlib.metadata import version
import torch

for package in ("torch", "transformers", "datasets"):
    print(f"{package} {version(package)}")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("This recipe requires a CUDA GPU with BF16 support")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PYENV
}

prepare() {
    hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir "$target"
    CONFIG="$cfg" bash examples/disagg/run_offline.sh --plan
    python examples/mmflash/prepare_chartqa.py --model "$target" --out "$data"
}

score() {
    python scripts/score_visual_kl.py --config "$cfg" --input "$data/train.jsonl" --scores "$cache/scores.jsonl"
}

parity() {
    if [[ ! -d "$author/.git" ]]; then
        git clone --filter=blob:none --no-checkout https://github.com/fscdc/MMFlash.git "$author"
        git -C "$author" sparse-checkout set specforge configs
        git -C "$author" checkout --detach "$pin"
    fi
    python scripts/gates/check_mmflash_parity.py --author-repo "$author" --target "$target" \
        --input "$data/train.jsonl" --scores "$cache/scores.jsonl" --features-dir "$cache/features" \
        --max-length 1024 --rows 2 --require-visual-signal --report "$out/parity.json"
}

train_arm() {
    local method="$1"
    local dir="$out/$method"
    # 64 steps at batch 1 over two epochs require at least 32 usable rows.
    python -c 'from pathlib import Path; n=sum(1 for _ in Path("cache/mmflash/features").glob("*.ckpt")); assert n >= 32, f"Need at least 32 captured rows for 64 steps, got {n}"'
    if [[ -e "$dir" ]]; then
        echo "Existing run directory: $dir. Use a fresh directory for a new run." >&2
        return 1
    fi
    local -a objective=()
    if [[ "$method" == dflash ]]; then
        objective=(training.objective=dflash training.objective_options.full_block_anchors=true
                   training.objective_options.loss_decay_gamma=7)
    fi
    CONFIG="$cfg" bash examples/disagg/run_offline.sh "${objective[@]}" "run_id=$method" "output_dir=$dir" \
        2>&1 | tee "$out/$method-train.log"
    python -m specforge.cli export --converter sglang --checkpoint "$dir/$method-step64" \
        --target-model "$target" --output-dir "$dir/export"
}

train() {
    train_arm dflash
    train_arm mmflash
}

serve_and_measure() (
    local method="$1"
    local draft="$out/$method/export"
    local log="$out/$method-server.log"
    # Fail before launching if some other service is already bound to this port.
    python -c 'import socket, sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$port"
    # setsid gives each server and its workers their own process group on Linux.
    setsid env SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 "${MMFLASH_SERVING_PYTHON:-python}" -m sglang.launch_server \
        --model-path "$target" --speculative-algorithm DFLASH --speculative-draft-model-path "$draft" \
        --speculative-dflash-block-size 16 --attention-backend triton --mm-attention-backend sdpa \
        --disable-radix-cache --dtype bfloat16 --context-length 2048 --max-running-requests 1 \
        --cuda-graph-max-bs 1 --stream-interval 1 --host 127.0.0.1 --port "$port" >"$log" 2>&1 &
    local server_pid=$!
    stop_server() {
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    }
    trap stop_server EXIT
    for _ in {1..360}; do
        if curl -fsS --max-time 3 "$url/health" >/dev/null 2>&1; then
            python examples/mmflash/benchmark.py run --url "$url" --draft "$draft" \
                --input "$data/heldout.jsonl" --out "$out/$method-bench.json"
            return
        fi
        kill -0 "$server_pid" 2>/dev/null || { tail -40 "$log" >&2; return 1; }
        sleep 3
    done
    echo "SGLang startup timed out; see $log" >&2
    return 1
)

bench() {
    serve_and_measure dflash
    serve_and_measure mmflash
    python examples/mmflash/benchmark.py compare --dflash "$out/dflash-bench.json" \
        --mmflash "$out/mmflash-bench.json" | tee "$out/comparison.json"
}

case "${1:-}" in
    preflight) mkdir -p "$out"; preflight | tee "$out/environment.txt" ;;
    dflash|mmflash) mkdir -p "$out"; train_arm "$1" ;;
    prepare|score|parity|train|bench) mkdir -p "$cache" "$out"; "$1" ;;
    all) mkdir -p "$cache" "$out"; preflight | tee "$out/environment.txt"; prepare; score; parity; train; bench ;;
    *) echo "Usage: $0 {preflight|prepare|score|parity|train|dflash|mmflash|bench|all}" >&2; exit 2 ;;
esac
