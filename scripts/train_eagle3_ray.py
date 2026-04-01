"""
Ray-based Eagle3 training entry point.

Usage
-----
python scripts/train_eagle3_ray.py \\
    --target-model-path <path> \\
    --train-data-path  <path> \\
    --output-dir       <path> \\
    [--disaggregate --rollout-num-gpus N --train-num-gpus M] \\
    [--rollout-tp-size R] \\
    [--train-tp-size T] \\
    [--train-sp-ulysses-size S] \\
    ...

Unlike the torchrun-based train_eagle3.py, this script does NOT use
torchrun / deepspeed as a launcher.  It starts Ray directly and
delegates all GPU compute to RolloutWorker / TrainWorker Actors.
"""

import argparse
import logging
import os
import sys

# Ensure the repo root is on sys.path when running directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from accelerate.utils import set_seed

from specforge.args import DisaggregateArgs, RayArgs, SGLangBackendArgs, TrackerArgs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Eagle3 with Ray-managed rollout and training workers."
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help="Draft model config path. If not provided, auto-generates from target model.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key of the embedding weight to load from the target model.",
    )
    model_group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM."
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="Backend for the target model.",
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Input data already has the chat template applied.",
    )
    dataset_group.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="Only the last assistant turn contributes to the loss.",
    )
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=4)

    # ── Training hyper-parameters ──────────────────────────────────────────
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument(
        "--max-num-steps",
        type=int,
        default=None,
        help="Maximum total training steps. Overrides num_epochs if set.",
    )
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. Auto-computed from epochs × steps_per_epoch if not set.",
    )
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="Test-Time Training (TTT) unroll length.",
    )
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory to start training from.",
    )
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument("--log-interval", type=int, default=50)
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    # ── Optimisation / parallelism ─────────────────────────────────────────
    opt_group = parser.add_argument_group("optimization")
    opt_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="(Legacy) TP size passed to SGLang backend. Use --rollout-tp-size instead.",
    )
    opt_group.add_argument("--sp-ulysses-size", type=int, default=1)
    opt_group.add_argument("--sp-ring-size", type=int, default=1)
    opt_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        help="Attention backend for the draft model (sdpa / fa / flex_attention / usp).",
    )

    # ── Output / cache ─────────────────────────────────────────────────────
    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="NCCL collective timeout in minutes.",
    )
    other_group.add_argument("--model-download-dir", type=str, default=None)

    # ── VLM ────────────────────────────────────────────────────────────────
    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument("--min-pixels", type=int, default=50176)
    vlm_group.add_argument("--max-pixels", type=int, default=802816)

    # ── SGLang backend ─────────────────────────────────────────────────────
    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    # ── Tracker ────────────────────────────────────────────────────────────
    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    # ── Ray ────────────────────────────────────────────────────────────────
    ray_group = parser.add_argument_group("ray")
    RayArgs.add_args(ray_group)

    # ── Disaggregated rollout ──────────────────────────────────────────────
    disagg_group = parser.add_argument_group("disaggregated rollout")
    DisaggregateArgs.add_args(disagg_group)

    args = parser.parse_args()

    # Back-fill legacy aliases so existing code paths still work
    if not hasattr(args, "target_batch_size"):
        rollout_tp = getattr(args, "rollout_tp_size", args.tp_size)
        args.target_batch_size = rollout_tp * args.batch_size

    return args


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(args.seed)

    # Validate disaggregate args
    disagg = DisaggregateArgs.from_args(args)
    disagg.validate(args)
    from specforge.ray_trainer.orchestrator import Eagle3RayOrchestrator
    orchestrator = Eagle3RayOrchestrator(args)
    try:
        orchestrator.run()
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
