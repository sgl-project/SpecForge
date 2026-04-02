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


def _add_model_args(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--target-model-path", type=str, required=True)
    group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    group.add_argument(
        "--draft-model-config",
        type=str,
        default=None,
        help="Draft model config path. If not provided, auto-generates from target model.",
    )
    group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="Key of the embedding weight to load from the target model.",
    )
    group.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="Key of the lm_head weight to load from the target model (DFlash only).",
    )
    group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM."
    )
    group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="Backend for the target model.",
    )
    group.add_argument(
        "--method",
        type=str,
        default="eagle3",
        choices=["eagle3", "dflash"],
        help="Training method: eagle3 (TTT unrolling) or dflash (block-parallel).",
    )


def _add_dataset_args(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--train-data-path", type=str, required=True)
    group.add_argument("--eval-data-path", type=str, default=None)
    group.add_argument("--chat-template", type=str, default="llama3")
    group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Input data already has the chat template applied.",
    )
    group.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="Only the last assistant turn contributes to the loss.",
    )
    group.add_argument("--build-dataset-num-proc", type=int, default=8)
    group.add_argument("--dataloader-num-workers", type=int, default=4)


def _add_training_args(parser):
    group = parser.add_argument_group("training")
    group.add_argument("--num-epochs", type=int, default=10)
    group.add_argument(
        "--max-num-steps",
        type=int,
        default=None,
        help="Maximum total training steps. Overrides num_epochs if set.",
    )
    group.add_argument("--batch-size", type=int, default=1)
    group.add_argument("--learning-rate", type=float, default=1e-4)
    group.add_argument("--max-length", type=int, default=2048)
    group.add_argument("--warmup-ratio", type=float, default=0.015)
    group.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. Auto-computed from epochs × steps_per_epoch if not set.",
    )
    group.add_argument("--max-grad-norm", type=float, default=0.5)
    group.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="Test-Time Training (TTT) unroll length.",
    )
    group.add_argument("--resume", action="store_true")
    group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint directory to start training from.",
    )
    group.add_argument("--eval-interval", type=int, default=5000)
    group.add_argument("--save-interval", type=int, default=5000)
    group.add_argument("--log-interval", type=int, default=50)
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--draft-accumulation-steps", type=int, default=1)


def _add_dflash_args(parser):
    group = parser.add_argument_group("dflash")
    group.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size for DFlash parallel generation.",
    )
    group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence (DFlash).",
    )
    group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Exponential loss decay gamma (DFlash). None = no decay.",
    )
    group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID for DFlash noise input. Auto-detected if not set.",
    )
    group.add_argument(
        "--num-draft-layers",
        type=int,
        default=5,
        help="Number of draft model layers (DFlash auto-config only).",
    )


def _add_optimization_args(parser):
    group = parser.add_argument_group("optimization")
    group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="(Legacy) TP size passed to SGLang backend. Use --rollout-tp-size instead.",
    )
    group.add_argument("--sp-ulysses-size", type=int, default=1)
    group.add_argument("--sp-ring-size", type=int, default=1)
    group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        help="Attention backend for the draft model (sdpa / fa / flex_attention / usp).",
    )


def _add_misc_args(parser):
    group = parser.add_argument_group("others")
    group.add_argument("--cache-dir", type=str, default="./cache")
    group.add_argument("--output-dir", type=str, required=True)
    group.add_argument("--verbose", action="store_true")
    group.add_argument(
        "--enable-perf",
        action="store_true",
        help="Print per-step timing breakdown for performance analysis.",
    )
    group.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="NCCL collective timeout in minutes.",
    )
    group.add_argument("--model-download-dir", type=str, default=None)

    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument("--min-pixels", type=int, default=50176)
    vlm_group.add_argument("--max-pixels", type=int, default=802816)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Eagle3/DFlash with Ray-managed rollout and training workers."
    )

    _add_model_args(parser)
    _add_dataset_args(parser)
    _add_training_args(parser)
    _add_dflash_args(parser)
    _add_optimization_args(parser)
    _add_misc_args(parser)

    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    ray_group = parser.add_argument_group("ray")
    RayArgs.add_args(ray_group)

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
    from specforge.ray_trainer.orchestrator import RayOrchestrator

    orchestrator = RayOrchestrator(args)
    try:
        orchestrator.run()
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
