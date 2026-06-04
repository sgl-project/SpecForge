import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from datasets import Dataset
from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_draft_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.target import (
    Eagle3TargetModel,
    TargetHead,
    get_eagle3_target_model,
)
from specforge.optimizer import BF16Optimizer
from specforge.tracker import Tracker, create_tracker, get_tracker_class
from specforge.utils import (
    create_draft_config_from_target,
    get_last_checkpoint,
    padding,
    print_args_with_dots,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
    safe_conversations_generator,
)


def parse_args() -> Tuple[ArgumentParser, Namespace]:
    """
    This function is used to parse the arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--draft-model-config",
        type=str,
        required=False,
        help="Draft model config path. If not provided, will auto-generate from target model.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model, this is only required for offline training",
    )
    model_group.add_argument(
        "--load-mtp-weights",
        action="store_true",
        help="Load MoE weights from the target model's native MTP head to initialize the draft model. "
        "Only applicable when the draft model uses MoE architecture (e.g., Qwen3MoeForCausalLMEagle3).",
    )
    model_group.add_argument(
        "--mtp-layer-idx",
        type=int,
        default=0,
        help="Index of the MTP block to load weights from (default: 0).",
    )
    model_group.add_argument(
        "--is-vlm", action="store_true", help="Whether the target model is a VLM"
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf", "custom"],
        help="The backend of the target model",
    )

    # dataset arguments
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--train-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-hidden-states-path", type=str, default=None)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="llama3")
    dataset_group.add_argument(
        "--is-preformatted",
        action="store_true",
        help="Whether the input data is preformatted text with the chat template already applied to the conversation messages.",
    )
    dataset_group.add_argument(
        "--train-only-last-turn",
        action="store_true",
        help="If set, only the last assistant turn in each conversation contributes to the loss. "
        "Useful for thinking models where conversation history may lack thought processes.",
    )
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=8)
    dataset_group.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    # training hyper params
    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=10)
    training_group.add_argument(
        "--max-num-steps",
        type=int,
        default=None,
        help="The maximum number of steps to train. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=1e-4)
    training_group.add_argument("--max-length", type=int, default=2048)
    training_group.add_argument("--warmup-ratio", type=float, default=0.015)
    training_group.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    training_group.add_argument("--max-grad-norm", type=float, default=0.5)
    training_group.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    training_group.add_argument(
        "--ploss-weight-strategy",
        type=str,
        default="decay",
        choices=["decay", "uniform", "reverse_linear", "custom"],
        help="How to weight per-step pLoss across `ttt_length` steps. "
        "'decay'=0.8**i (legacy default, biased to step 0); "
        "'uniform'=all 1; "
        "'reverse_linear'=[0.5, 1.0, 1.5, ...] (favors later steps, "
        "useful when later TTT steps under-train, e.g. MTP init + ttt>1); "
        "'custom'=use --ploss-weights.",
    )
    training_group.add_argument(
        "--ploss-weights",
        type=str,
        default=None,
        help="Comma-separated per-step weights, used only when "
        "--ploss-weight-strategy=custom. Length MUST equal --ttt-length. "
        "Example for ttt_length=4: '0.5,1.0,1.5,2.0'.",
    )
    training_group.add_argument(
        "--normalize-ploss-weights",
        action="store_true",
        help="If set, rescale weights so that sum(w) == ttt_length (i.e. "
        "average weight == 1). This keeps the *total* gradient magnitude "
        "comparable to the uniform baseline, so you do NOT have to retune "
        "learning rate when switching strategies. Strongly recommended.",
    )
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="directory includes the checkpoint to start training with",
    )
    training_group.add_argument("--eval-interval", type=int, default=5000)
    training_group.add_argument("--save-interval", type=int, default=5000)
    training_group.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )
    training_group.add_argument("--seed", type=int, default=0)
    training_group.add_argument("--draft-accumulation-steps", type=int, default=1)

    # data processing type
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )
    optimization_group.add_argument(
        "--target-micro-batch-size",
        type=int,
        default=0,
        help="Number of samples per micro-batch for target model inference. "
        "When > 0 and vocab mapping is available, enables memory-optimized mode: "
        "1) processes samples in micro-batches to reduce peak memory, "
        "2) projects full-vocab logits to draft vocab immediately after each micro-batch. "
        "Recommended value: 1 (minimum memory). 0 = process all samples at once (original behavior). "
        "This is essential for large-vocab models (e.g., 248K vocab) with long sequences (e.g., 20K tokens).",
    )
    optimization_group.add_argument(
        "--logits-chunk-size",
        type=int,
        default=2048,
        help="Chunk size (in tokens) for chunked logits computation during target model inference. "
        "When > 0 and total tokens exceed this value, logits are computed in chunks to reduce "
        "the peak memory of tensor_model_parallel_all_gather from (total_tokens × vocab_size) "
        "to (chunk_size × vocab_size). For chunk_size=2048 and vocab_size=248K: peak ≈ 0.96 GB. "
        "0 = disable chunking (compute all logits at once, original behavior).",
    )
    # distributed training
    optimization_group.add_argument("--sp-ulysses-size", type=int, default=1)
    optimization_group.add_argument("--sp-ring-size", type=int, default=1)
    optimization_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        help="The attention backend for the draft model",
    )

    # other args
    other_group = parser.add_argument_group("others")
    other_group.add_argument("--cache-key", type=str, default=None)
    other_group.add_argument("--cache-dir", type=str, default="./cache")
    other_group.add_argument("--output-dir", type=str, required=True)
    other_group.add_argument("--verbose", action="store_true")
    other_group.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    other_group.add_argument(
        "--model-download-dir",
        type=str,
        default=None,
        help="The directory to download the target model to",
    )

    # vlm related args
    vlm_group = parser.add_argument_group("vlm")
    vlm_group.add_argument(
        "--min-pixels", type=int, default=50176
    )  # 64*28*28 for qwen2.5-vl
    vlm_group.add_argument(
        "--max-pixels", type=int, default=802816
    )  # 1024*28*28 for qwen2.5-vl

    # profiling related args
    profiling_group = parser.add_argument_group("profiling")
    profiling_group.add_argument("--profile", action="store_true")
    profiling_group.add_argument("--profile-start-step", type=int, default=30)
    profiling_group.add_argument("--profile-num-steps", type=int, default=4)
    profiling_group.add_argument("--profile-record-shapes", action="store_true")

    # sglang target model backend related args
    sglang_group = parser.add_argument_group("sglang target model backend")
    SGLangBackendArgs.add_args(sglang_group)

    # tracker related args
    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    args = parser.parse_args()
    return parser, args


def build_tracker(args: Namespace, parser: ArgumentParser) -> Tracker:
    """
    Build the experiment tracker according to the report_to argument.

    Args:
        args: The arguments for the training script.
        parser: The parser for the training script.

    Returns:
        The experiment tracker.
    """
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    return tracker


def build_target_model(
    args: Namespace, draft_model_config: AutoDraftModelConfig, is_online: bool = True
) -> Tuple[Union[Eagle3TargetModel, TargetHead], Optional[AutoProcessor]]:
    """
    Build the target model according to the arguments.

    Args:
        args: The arguments for the training script.
        draft_model_config: The draft model config.

    Returns:
        The target model.
    """
    if is_online:
        if (
            args.is_vlm
            and draft_model_config.target_model_type == "qwen2_5_vl"
            and args.target_model_backend == "custom"
        ):
            from transformers import Qwen2_5_VLForConditionalGeneration

            target_model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
        else:
            if args.target_model_backend == "sglang":
                target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
            else:
                target_model_kwargs = {}
            target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=args.target_model_path,
                backend=args.target_model_backend,
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.model_download_dir,
                **target_model_kwargs,
                trust_remote_code=args.trust_remote_code,
            )

        # set the aux hidden states layers
        if args.load_mtp_weights:
            # MTP single-hidden-state regime: the draft consumes ONE hidden state
            # from the target's last transformer layer (matching how the target's
            # native MTP head is wired in inference). We therefore configure the
            # target to capture exactly one layer — the last one — so that the
            # tensor reaching OnlineEagle3Model.forward has shape [B, S, H] and
            # `project_hidden_states` can stay an identity pass-through.
            from transformers import AutoConfig as _AutoConfig

            _t_cfg = _AutoConfig.from_pretrained(
                args.target_model_path,
                trust_remote_code=args.trust_remote_code,
            )
            # Multimodal targets nest the LM config under `text_config`.
            _text_cfg = getattr(_t_cfg, "text_config", _t_cfg)
            if not hasattr(_text_cfg, "num_hidden_layers"):
                raise ValueError(
                    "Cannot determine target's num_hidden_layers for MTP mode "
                    "from config; expected `num_hidden_layers` on either the "
                    "root config or `text_config`."
                )
            _last_layer_idx = _text_cfg.num_hidden_layers - 1
            print_on_rank0(
                f"[MTP] Configuring target to capture single hidden state "
                f"from last transformer layer (idx={_last_layer_idx})."
            )
            target_model.set_aux_hidden_states_layers([_last_layer_idx])
        elif (
            hasattr(draft_model_config, "eagle_config")
            and draft_model_config.eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
        ):
            target_model.set_aux_hidden_states_layers(
                draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
            )
        else:
            target_model.set_aux_hidden_states_layers()

        if args.is_vlm:
            processor = AutoProcessor.from_pretrained(
                args.target_model_path,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        else:
            processor = None

        return target_model, processor
    else:
        target_head = TargetHead.from_pretrained(
            model_path=args.target_model_path,
            lm_head_key=args.lm_head_key,
            cache_dir=args.model_download_dir,
            trust_remote_code=args.trust_remote_code,
        )
        return target_head, None


def sanity_check(args: Namespace) -> None:
    """
    Perform sanity checks on the arguments.

    Args:
        args: The arguments for the training script.

    Returns:
        None
    """
    args.dp_size = dist.get_world_size() // args.tp_size
    args.target_batch_size = args.tp_size * args.batch_size
    if args.attention_backend == "usp":
        sp_sanity_check(args)


def sp_sanity_check(args: Namespace) -> None:
    args.draft_accumulation_steps = (
        args.draft_accumulation_steps * args.sp_ulysses_size * args.sp_ring_size
    )
    assert (
        args.batch_size == 1
    ), f"USP only supports batch_size=1, got batch_size={args.batch_size}"

    assert args.sp_ring_size * args.sp_ulysses_size > 1, (
        f"USP requires sp_ring_size * sp_ulysses_size > 1. "
        f"Got sp_ring_size={args.sp_ring_size}, sp_ulysses_size={args.sp_ulysses_size}."
    )

    assert args.train_hidden_states_path is not None, f"USP only support offline mode"

    if args.eval_data_path is not None and args.eval_hidden_states_path is not None:
        raise ValueError(
            "Cannot set both eval_data_path and eval_hidden_states_path. "
            "For online mode, set only eval_data_path. "
            "For offline mode, set only eval_hidden_states_path."
        )


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, nn.Module]:
    # ckpt info(epoch, step)
    ckpt_info = (0, 0)

    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, cache_dir=args.model_download_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

    # Handle base ckpt, config file
    draft_model_last_checkpoint = None
    is_resume_checkpoint = False
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_config = AutoDraftModelConfig.from_file(
                os.path.join(args.ckpt_dir, "config.json")
            )
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )

    # detecting last ckpt for draft model
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        is_resume_checkpoint = True

    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
        ).cuda()

    # Load training state (optimizer, scheduler, epoch, step) for true resume
    resume_state = None
    if is_resume_checkpoint and draft_model_last_checkpoint:
        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_rank0(
                f"Loaded training state from {training_state_path}: "
                f"epoch={resume_state['epoch']}, step={resume_state['global_step']}"
            )

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()

    # Load MTP weights for MoE draft models if requested.
    # Two paths:
    #   (a) Fresh start (no checkpoint) — copy ALL weights from target MTP head
    #       so the draft model is initialized as an exact replica.
    #   (b) Resume — checkpoint was saved WITHOUT lm_head (it is frozen and
    #       always equal to target's `mtp.layers.{idx}.shared_head.head`).
    #       Reload ONLY lm_head so the resumed model matches the original init.
    if args.load_mtp_weights and hasattr(draft_model, "load_mtp_weights"):
        if not draft_model_last_checkpoint:
            print_on_rank0(
                f"Loading MTP weights from {args.target_model_path} "
                f"(mtp_block_{args.mtp_layer_idx})"
            )
            draft_model.load_mtp_weights(
                args.target_model_path,
                mtp_layer_idx=args.mtp_layer_idx,
            )
        else:
            print_on_rank0(
                f"Resume detected: re-loading lm_head only from target MTP "
                f"(mtp_block_{args.mtp_layer_idx}) — other weights kept from checkpoint"
            )
            draft_model.load_mtp_weights(
                args.target_model_path,
                mtp_layer_idx=args.mtp_layer_idx,
                only_lm_head=True,
            )

    # MTP mode: freeze the (full-vocab) lm_head so it stays identical to
    # target's `shared_head.head`. This is what makes the trained draft
    # weights drop-in compatible with target's native MTP inference path.
    if args.load_mtp_weights and hasattr(draft_model, "freeze_lm_head"):
        draft_model.freeze_lm_head()
        print_on_rank0("Froze draft lm_head (MTP target-equivalence guarantee)")

    return draft_model_config, draft_model, ckpt_info, resume_state


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )

    # convert to dataloader
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = Dataset.from_generator(
        generator=safe_conversations_generator,
        gen_kwargs={"file_path": args.train_data_path},
    )
    is_online = (
        args.train_data_path is not None and args.train_hidden_states_path is None
    )
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            is_vlm=args.is_vlm,
            is_preformatted=args.is_preformatted,
            processor=processor,
            num_proc=args.build_dataset_num_proc,
            train_only_last_turn=args.train_only_last_turn,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )

        if not is_online:
            train_eagle3_dataset = build_offline_eagle3_dataset(
                args.train_hidden_states_path,
                args.max_length,
                ttt_length=args.ttt_length,
                use_usp_preprocess=(args.attention_backend == "usp"),
            )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=(
            get_draft_dp_group()
            if args.attention_backend == "usp" and not is_online
            else get_dp_group()
        ),
        is_vlm=args.is_vlm,
    )
    if args.eval_data_path is not None or args.eval_hidden_states_path is not None:
        if args.eval_data_path is not None:
            eval_dataset = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": args.eval_data_path},
            )
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                is_vlm=args.is_vlm,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
                train_only_last_turn=args.train_only_last_turn,
            )
        elif args.eval_hidden_states_path is not None:
            eval_eagle3_dataset = build_offline_eagle3_dataset(
                args.eval_hidden_states_path,
                args.max_length,
                ttt_length=args.ttt_length,
                use_usp_preprocess=(args.attention_backend == "usp"),
            )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=(
                get_draft_dp_group()
                if args.attention_backend == "usp" and not is_online
                else get_dp_group()
            ),
            is_vlm=args.is_vlm,
        )
        print_with_rank("Initialized eval dataloader")
    else:
        eval_dataloader = None
    return (
        train_dataloader,
        vocab_mapping_path,
        eval_dataloader,
    )


def save_checkpoints(
    args: Namespace,
    epoch: int,
    step: int,
    eagle3_model: nn.Module,
    optimizer: Optimizer,
):
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(epoch_output_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(eagle3_model, StateDictType.FULL_STATE_DICT):
        model_state_dict = eagle3_model.state_dict()
        state_to_save = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
        }
        state_to_save.update(optimizer.state_dict())
        # Filter out:
        #   - embed_tokens: shared with target (loaded from `target_model_path`
        #     via `load_embedding`), kept frozen.
        #     IMPORTANT: match by exact substring `embed_tokens` (NOT just
        #     `embed`), otherwise we will accidentally drop other learnable
        #     parameters whose names happen to contain the substring
        #     "embed", such as MTP's `pre_fc_norm_embedding.weight`.
        #     Historical bug: the old check `"embed" not in k.lower()` silently
        #     dropped `pre_fc_norm_embedding` from MTP checkpoints, leading to
        #     a missing key on resume + degraded sglang acceptance length.
        #   - lm_head: in MTP mode this is also frozen and identical to
        #     target's `shared_head.head`; on resume we reload it from
        #     `target_model_path` (see `build_draft_model`). For Eagle3 mode
        #     the draft lm_head is shaped (draft_vocab, hidden) and is small
        #     enough that the filter still saves ~tens of MB without harm
        #     since `from_pretrained` will simply re-init missing keys from
        #     the config (Eagle3 baseline already trains lm_head — keep it!).
        # To stay backwards-compatible with Eagle3, only filter lm_head when
        # it is frozen (i.e. MTP mode).
        lm_head_frozen = (
            args.load_mtp_weights
            and hasattr(eagle3_model.draft_model, "freeze_lm_head")
        )
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k
            and "embed_tokens" not in k
            and not (lm_head_frozen and "lm_head" in k)
        }

        if dist.get_rank() == 0:
            torch.save(
                state_to_save,
                os.path.join(epoch_output_dir, "training_state.pt"),
            )
            print_on_rank0(
                f"Saved full training state to {epoch_output_dir}/training_state.pt"
            )
            eagle3_model.draft_model.save_pretrained(
                epoch_output_dir,
                state_dict=draft_model_state_dict,
            )
            print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
        dist.barrier()


def run_forward(
    args: Namespace,
    eagle3_model: nn.Module,
    data: dict,
    target_model: Optional[Eagle3TargetModel] = None,
    is_online: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if args.is_vlm and args.target_model_backend == "custom":
        plosses, _, acces = eagle3_model(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
            pixel_values=data["pixel_values"].cuda(),
            image_grid_thw=data["image_grid_thw"].cuda(),
        )
    else:
        image_grid_thw = None
        pre_projected = False
        target_in_draft_mask = None
        if is_online:
            # we generate the eagle3 using the target model in an online fashion
            # Handle VLM data: pixel_values and image_grid_thw are lists
            target_mbs = getattr(args, "target_micro_batch_size", 0)
            if args.is_vlm:
                image_grid_thw = (
                    [thw.cuda().squeeze() for thw in data["image_grid_thw"]]
                    if args.is_vlm
                    else None
                )
                pixel_values = data["pixel_values"].cuda()
                eagle3_data = target_model.generate_eagle3_data(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    is_vlm=args.is_vlm,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    target_micro_batch_size=target_mbs,
                )
            else:
                eagle3_data = target_model.generate_eagle3_data(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                    target_micro_batch_size=target_mbs,
                )

            input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
            attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
            loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
            target = get_dp_data_shard_from_tp(eagle3_data.target)
            hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)
            pre_projected = eagle3_data.pre_projected
            if eagle3_data.target_in_draft_mask is not None:
                target_in_draft_mask = get_dp_data_shard_from_tp(
                    eagle3_data.target_in_draft_mask
                )

            # Apply padding AFTER DP sharding to avoid 2x peak memory.
            # Before sharding: target is (tp_size, seq_len, vocab) e.g. (8, 20480, 32000)
            # = 9.77 GiB.  padding() would need another 9.77 GiB copy → OOM.
            # After sharding: target is (1, seq_len, vocab) = 1.22 GiB → padding trivially fits.
            #
            # IMPORTANT: get_dp_data_shard_from_tp returns a view (via chunk), so the
            # original (8, ...) tensor stays alive until all views are released.
            # We must .contiguous() the large tensors to create independent copies,
            # then del eagle3_data to free the original 9.77 GiB tensor before padding.
            target = target.contiguous()
            hidden_states = hidden_states.contiguous()
            if target_in_draft_mask is not None:
                target_in_draft_mask = target_in_draft_mask.contiguous()
            del eagle3_data
            torch.cuda.empty_cache()

            input_ids = padding(input_ids, left=False)
            target = padding(target, left=False)
            if target_in_draft_mask is not None:
                target_in_draft_mask = padding(target_in_draft_mask, left=False)
        else:
            # we generate the logits using the hidden states loaded from disk
            if data["hidden_state"] is None:
                # Corrupt batch: prepare_hidden_states.py left a sample without
                # aux_hidden_state (silent prepare-time IO race, ~1/96k probability
                # observed on 96277-sample dataset). Return zero-loss/zero-acc
                # placeholders so training continues; this batch will produce
                # zero gradient and the optimizer step is effectively a no-op.
                device = next(eagle3_model.parameters()).device
                ttt_length = getattr(args, "ttt_length", 4)
                if torch.distributed.get_rank() == 0:
                    print(
                        f"[WARN] skipping batch with None hidden_state "
                        f"(corrupt prepare-time data)"
                    )
                zero = torch.zeros(1, device=device, dtype=torch.bfloat16)
                plosses = [zero.clone().requires_grad_(True) for _ in range(ttt_length)]
                acces = [zero.clone() for _ in range(ttt_length)]
                return plosses, acces
            attention_mask = data["attention_mask"].cuda()
            hidden_states = data["hidden_state"].cuda()
            input_ids, target, loss_mask = target_model.preprocess(
                data["input_ids"], data["target"], data["loss_mask"]
            )
            input_ids = input_ids.cuda()
            # target_model(hidden_states) returns:
            #   - With t2d_mapping: (projected_logits, target_in_draft_mask) — draft vocab
            #   - Without t2d_mapping: (full_logits, None) — full vocab
            target_result = target_model(target.cuda())
            if isinstance(target_result, tuple):
                target, target_in_draft_mask = target_result
                if target_in_draft_mask is not None:
                    pre_projected = True
            else:
                # Backward compatibility: old code returned a single tensor
                target = target_result
            loss_mask = loss_mask.cuda()
        plosses, _, acces = eagle3_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target=target,
            hidden_states=hidden_states,
            position_ids=(
                data["position_ids"].cuda() if "position_ids" in data else None
            ),
            image_grid_thw=image_grid_thw,
            is_vlm=args.is_vlm,
            pre_projected=pre_projected,
            target_in_draft_mask=target_in_draft_mask,
        )
    return plosses, acces


def _resolve_ploss_weight(args: Namespace, n: int) -> List[float]:
    """Compute the per-step pLoss weight vector of length `n` (= ttt_length).

    Selection order:
      - "uniform"        -> [1.0] * n
      - "reverse_linear" -> [0.5, 1.0, 1.5, ..., 0.5*(n)]   (later steps weighted higher)
      - "custom"         -> parsed from --ploss-weights, must have length n
      - "decay" (legacy) -> [0.8 ** i for i in range(n)]    (earlier steps weighted higher)

    If --normalize-ploss-weights is set, the weights are rescaled so that
    sum(w) == n (i.e. average weight == 1). This keeps the *total* gradient
    magnitude comparable to the uniform baseline, so the effective learning
    rate stays roughly the same when switching strategies.
    """
    strategy = getattr(args, "ploss_weight_strategy", "decay")
    if strategy == "uniform":
        weights = [1.0] * n
    elif strategy == "reverse_linear":
        # 0.5, 1.0, 1.5, 2.0, ...  (linearly increasing, never zero at step 0)
        weights = [0.5 * (i + 1) for i in range(n)]
    elif strategy == "custom":
        raw = getattr(args, "ploss_weights", None)
        assert raw, (
            "--ploss-weights is required when --ploss-weight-strategy=custom"
        )
        weights = [float(x) for x in raw.split(",")]
        assert len(weights) == n, (
            f"len(--ploss-weights)={len(weights)} does not match "
            f"--ttt-length={n}; please provide exactly {n} comma-separated values."
        )
    else:  # "decay" — legacy default
        weights = [0.8**i for i in range(n)]

    if getattr(args, "normalize_ploss_weights", False):
        s = sum(weights)
        assert s > 0, f"sum(ploss_weights)={s} is non-positive; cannot normalize."
        weights = [n * w / s for w in weights]

    return weights


def run_backward_and_update(
    args: Namespace, plosses: List[torch.Tensor], optimizer: Optimizer, global_step: int
) -> None:
    ploss_weight = _resolve_ploss_weight(args, len(plosses))
    ploss = (
        sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        / args.draft_accumulation_steps
    )
    ploss.backward()

    if global_step % args.draft_accumulation_steps == 0:
        optimizer.step()


def record_metrcs(
    args: Namespace,
    accuracies: List[torch.Tensor],
    plosses: List[torch.Tensor],
    global_step: int,
    tracker: Tracker,
    optimizer: Optional[Optimizer] = None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    accuracies = torch.stack(accuracies)
    plosses = torch.stack(plosses)

    assert accuracies.shape[0] == args.ttt_length
    dist.all_reduce(accuracies, op=dist.ReduceOp.AVG)
    accuracies = accuracies.cpu().tolist()
    for i in range(len(accuracies)):
        logdict[f"{mode}/acc_{i}"] = accuracies[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i},  Acc: {accuracies[i]:.2f}"
        )

    dist.all_reduce(plosses, op=dist.ReduceOp.AVG)
    plosses = plosses.cpu().tolist()
    for i in range(len(plosses)):
        logdict[f"{mode}/ploss_{i}"] = plosses[i]
        print_on_rank0(
            f"Eval - Step {global_step} [{global_step + 1}/{args.num_epochs}], position {i}, pLoss: {plosses[i]}"
        )
    tracker.log(logdict, step=global_step)


def get_dp_data_shard_from_tp(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get the data shard from the tensor.
    """
    tp_size = dist.get_world_size(get_tp_group())
    tp_rank = dist.get_rank(get_tp_group())
    return tensor.chunk(tp_size, dim=0)[tp_rank]


def main():
    # ================================================
    # 1. Initialize
    # ================================================
    parser, args = parse_args()
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    is_online = (
        args.train_data_path is not None and args.train_hidden_states_path is None
    )

    sanity_check(args)
    print_args_with_dots(args)
    print_with_rank("Initialized distributed environment")

    # ================================================
    # 2. Build models
    # ================================================
    draft_model_config, draft_model, ckpt_info, resume_state = build_draft_model(args)
    target_model, processor = build_target_model(args, draft_model_config, is_online)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config, processor
    )

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    # Pass vocab mapping to target model for early projection (memory optimization).
    # This is critical for large-vocab models (e.g. 248K vocab): without it, the full-
    # vocab logits tensor (~7.7 GiB for 16K tokens) would cause OOM.
    # Early projection reduces this to ~1.0 GiB (draft_vocab=32K).
    # Works for both online mode (SGLang logits processor) and offline mode (TargetHead).
    if hasattr(draft_model, "t2d") and hasattr(target_model, "set_vocab_mapping"):
        target_model.set_vocab_mapping(draft_model.t2d)
        mode_str = "online" if is_online else "offline"
        print_with_rank(
            f"Early vocab projection enabled ({mode_str}, full_vocab → draft_vocab)"
        )

    # Set chunked logits computation size (memory optimization)
    if hasattr(target_model, "set_logits_chunk_size"):
        target_model.set_logits_chunk_size(args.logits_chunk_size)
        if args.logits_chunk_size > 0:
            vocab_size = getattr(draft_model.config, "vocab_size", 248320)
            print_with_rank(
                f"Enabled chunked logits: chunk_size={args.logits_chunk_size} tokens "
                f"(peak ≈ {args.logits_chunk_size * vocab_size * 2 / 1024**3:.2f} GB per chunk)"
            )
        else:
            print_with_rank("Chunked logits disabled (--logits-chunk-size 0)")

    # Draft-side chunked lm_head (memory optimization for full-vocab MTP head).
    # The MTP draft has a 248K-vocab lm_head; computing logits in chunks and
    # immediately t2d-projecting to draft_vocab keeps peak activation memory
    # bounded. Only available on MTP-compatible draft models.
    if hasattr(draft_model, "set_lm_head_chunk_size") and args.logits_chunk_size > 0:
        draft_model.set_lm_head_chunk_size(args.logits_chunk_size)
        print_with_rank(
            f"Draft lm_head chunked: chunk_size={args.logits_chunk_size} tokens"
        )

    # Calculate total steps if not provided
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} (num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    else:
        print_with_rank(f"Using provided total_steps: {args.total_steps}")

    # ================================================
    # 4. Build Eagle3 model
    # ================================================
    if (
        args.is_vlm
        and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl"
        and args.tp_size == 1
        and args.target_model_backend != "sglang"
    ):
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        if is_online:
            eagle3_model = OnlineEagle3Model(
                target_model=target_model,
                draft_model=draft_model,
                length=args.ttt_length,
                attention_backend=args.attention_backend,
            )
        else:
            # offline: the target_model is TargetHead not a model
            eagle3_model = OnlineEagle3Model(
                draft_model=draft_model,
                length=args.ttt_length,
                attention_backend=args.attention_backend,
            )
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,  # the draft model should run dp for all processes
    )
    print_with_rank("Initialized Eagle3 FSDP model")

    # ================================================
    # 5. Build optimizer and scheduler
    # ================================================
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")

    # Restore optimizer/scheduler state for true resume
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        print_on_rank0(
            f"Restored optimizer/scheduler state: "
            f"epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )
        del resume_state
    else:
        start_epoch = ckpt_info[0]
        global_step = ckpt_info[1]

    # Calculate how many steps to skip in the current epoch (for dataloader fast-forward)
    skip_steps = global_step - start_epoch * len(train_dataloader)

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 7. Start training
    # ================================================
    print_on_rank0(
        f"Starting training from epoch:{start_epoch}          step:{global_step}"
    )

    # Resolve and log the effective per-step pLoss weights once for transparency.
    # Doing this here (before the train loop) makes the chosen strategy
    # immediately visible in the rank-0 log, which is helpful when comparing runs.
    _effective_ploss_weight = _resolve_ploss_weight(args, args.ttt_length)
    print_on_rank0(
        f"[pLoss] strategy={args.ploss_weight_strategy}, "
        f"normalize={args.normalize_ploss_weights}, "
        f"weights={['%.4f' % w for w in _effective_ploss_weight]} "
        f"(sum={sum(_effective_ploss_weight):.4f}, ttt_length={args.ttt_length})"
    )

    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            # Skip steps already processed in the current epoch when resuming
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue

            global_step += 1

            # ================================================
            # 7.0 Profiling
            # ================================================
            if args.profile:
                # we add the step by 1 to align with global step
                if global_step == args.profile_start_step + 1:
                    print("Start profile")
                    torch_profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        with_stack=True,
                        record_shapes=args.profile_record_shapes,
                    )
                    torch_profiler.start()
                if global_step == args.profile_start_step + args.profile_num_steps + 1:
                    output_path = os.path.join(
                        args.output_dir,
                        f"profile_rank{torch.distributed.get_rank()}_{time.time()}.trace.json.gz",
                    )
                    print(f"End profile {output_path=}")
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(output_path)

            # ================================================
            # 7.1 Training Step
            # ================================================
            plosses, acces = run_forward(
                args,
                eagle3_model,
                data,
                target_model,
                is_online,
            )
            run_backward_and_update(args, plosses, optimizer, global_step)

            # log training metrics
            if global_step % (args.log_interval * args.draft_accumulation_steps) == 0:
                record_metrcs(
                    args,
                    acces,
                    plosses,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    optimizer,
                    mode="train",
                )

            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                avg_loss = sum(pl for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.2f}",
                        "acc": f"{avg_acc:.2f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )

            # ================================================
            # 7.2 Evaluation Step
            # ================================================
            should_evaluate = (
                args.eval_data_path is not None
                or args.eval_hidden_states_path is not None
            )
            if (
                should_evaluate
                and global_step % (args.eval_interval * args.draft_accumulation_steps)
                == 0
            ):
                # Run evaluation
                draft_model.eval()
                eval_acces = [[] for _ in range(eagle3_model.length)]
                eval_plosses = [[] for _ in range(eagle3_model.length)]

                for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    with torch.no_grad():
                        plosses, acces = run_forward(
                            args, eagle3_model, data, target_model, is_online
                        )
                        eval_acces = [
                            eval_acces[i] + [acces[i]] for i in range(len(acces))
                        ]
                        eval_plosses = [
                            eval_plosses[i] + [plosses[i]] for i in range(len(plosses))
                        ]

                # compute average over all minibatches
                eval_acces = [torch.stack(acc).mean() for acc in eval_acces]
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]

                record_metrcs(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step // args.draft_accumulation_steps,
                    tracker,
                    mode="eval",
                )
            # ================================================
            # 7.3 Save Checkpoints
            # ================================================
            if global_step % args.save_interval == 0:
                # Save the model
                save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break

        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break
    # Save final checkpoint if training ended without saving
    if global_step % args.save_interval != 0:
        print_on_rank0(
            f"Training completed at step {global_step}, saving final checkpoint..."
        )
        save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

    # Close the tracker
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
