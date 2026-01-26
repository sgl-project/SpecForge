import argparse
import hashlib
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import Dataset
from specforge import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    MiniCPMVLOnlineEagle3Model,
    OnlineEagle3Model,
    QwenVLOnlineEagle3Model,
)
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.data import (
    build_eagle3_dataset,
    build_offline_eagle3_dataset,
    generate_vocab_mapping_file,
    generate_vocab_mapping_for_vlm,
    prepare_dp_dataloaders,
    # New VLM Template architecture
    build_vlm_dataset_with_template,
    get_vlm_template,
    prepare_vlm_dataloader,
)
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_draft_dp_group,
    get_draft_sp_group,
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
        "--save-per-epoch",
        action="store_true",
        help="Save checkpoint at the end of each epoch",
    )
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
        "--debug-first-sample",
        action="store_true",
        help="Print first training sample details for debugging",
    )
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
) -> Tuple[Union[Eagle3TargetModel, TargetHead], Optional[Any]]:
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
            and args.tp_size == 1
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
        elif (
            args.is_vlm
            and draft_model_config.target_model_type == "minicpm_v_4"
            and args.tp_size == 1
        ):
            from transformers import AutoModel

            target_model = (
                AutoModel.from_pretrained(
                    pretrained_model_name_or_path=args.target_model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
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

        # set the aux hidden states layers (not needed for MiniCPM-V which handles it internally)
        if draft_model_config.target_model_type != "minicpm_v_4":
            if (
                hasattr(draft_model_config, "eagle_config")
                and draft_model_config.eagle_config is not None
                and "eagle_aux_hidden_state_layer_ids" in draft_model_config.eagle_config
            ):
                target_model.set_aux_hidden_states_layers(
                    draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
                )
            else:
                target_model.set_aux_hidden_states_layers()

        # Load processor for MiniCPM-V (needed by MiniCPMVLOnlineEagle3Model)
        processor = None
        if args.is_vlm:
            from transformers import AutoProcessor
            trust_remote_code = draft_model_config.target_model_type == "minicpm_v_4"
            processor = AutoProcessor.from_pretrained(
                args.target_model_path,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                trust_remote_code=trust_remote_code,
            )

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
    args.draft_accumulation_steps = (
        args.draft_accumulation_steps * args.sp_ulysses_size * args.sp_ring_size
    )

    if args.eval_data_path is not None and args.eval_hidden_states_path is not None:
        raise ValueError(
            "Cannot set both eval_data_path and eval_hidden_states_path. "
            "For online mode, set only eval_data_path. "
            "For offline mode, set only eval_hidden_states_path."
        )


def build_draft_model(args: Namespace) -> Tuple[AutoDraftModelConfig, nn.Module]:
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
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()

    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    return draft_model_config, draft_model


def build_dataloaders(
    args: Namespace,
    draft_model_config: AutoDraftModelConfig,
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
    
    vlm_type = getattr(draft_model_config, "target_model_type", None)
    vlm_template = None
    
    if args.is_vlm:
        # VLM: Use VLM Template architecture
        print_with_rank(f"Using VLM Template architecture for {vlm_type}")
        
        # Generate vocab mapping FIRST using text-only processing (fast)
        with rank_0_priority():
            vocab_mapping_path = generate_vocab_mapping_for_vlm(
                hf_dataset=train_dataset,
                tokenizer_path=args.target_model_path,
                target_vocab_size=draft_model_config.vocab_size,
                draft_vocab_size=draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
                cache_key=cache_key,
                num_proc=args.build_dataset_num_proc,
            )
        
        # Then create VLM dataset with full image processing
        vlm_template = get_vlm_template(
            vlm_type=vlm_type,
            processor_path=args.target_model_path,
            max_length=args.max_length,
        )
        train_eagle3_dataset = build_vlm_dataset_with_template(
            dataset=train_dataset,
            vlm_template=vlm_template,
            shuffle_seed=42,
            num_proc=args.build_dataset_num_proc,
        )
        
        train_dataloader = prepare_vlm_dataloader(
            train_eagle3_dataset,
            vlm_template,
            args.target_batch_size,
            num_workers=0,  # Must be 0 for VLM Template
            shuffle=True,
            process_group=(
                get_draft_dp_group()
                if args.attention_backend == "usp" and not is_online
                else get_dp_group()
            ),
        )
    else:
        # LLM: Use original preprocessing
        with rank_0_priority():
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=cache_key,
                is_preformatted=args.is_preformatted,
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
        )

    if args.eval_data_path is not None or args.eval_hidden_states_path is not None:
        if args.eval_data_path is not None:
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
            if args.is_vlm:
                eval_eagle3_dataset = build_vlm_dataset_with_template(
                    dataset=eval_dataset,
                    vlm_template=vlm_template,
                    shuffle_seed=42,
                    num_proc=args.build_dataset_num_proc,
                )
                eval_dataloader = prepare_vlm_dataloader(
                    eval_eagle3_dataset,
                    vlm_template,
                    args.target_batch_size,
                    num_workers=0,
                    shuffle=False,
                    process_group=(
                        get_draft_dp_group()
                        if args.attention_backend == "usp" and not is_online
                        else get_dp_group()
                    ),
                )
            else:
                eval_eagle3_dataset = build_eagle3_dataset(
                    eval_dataset,
                    tokenizer,
                    args.chat_template,
                    args.max_length,
                    num_proc=args.build_dataset_num_proc,
                    is_preformatted=args.is_preformatted,
                    train_only_last_turn=args.train_only_last_turn,
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
                )
        elif args.eval_hidden_states_path is not None:
            eval_eagle3_dataset = build_offline_eagle3_dataset(
                args.eval_hidden_states_path,
                args.max_length,
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
        draft_model_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in model_state_dict.items()
            if "draft_model." in k and "embed" not in k.lower()
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
    vlm_type: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if args.is_vlm:
        if vlm_type == "minicpm_v_4":
            # MiniCPM-V: pixel_values is nested list [[tensor, ...], ...], move tensors to cuda
            def move_to_cuda(item):
                if item is None:
                    return None
                if isinstance(item, torch.Tensor):
                    return item.cuda()
                if isinstance(item, list):
                    return [move_to_cuda(x) for x in item]
                return item
            
            pixel_values = move_to_cuda(data["pixel_values"])
            tgt_sizes = move_to_cuda(data["tgt_sizes"])
            image_bound = move_to_cuda(data["image_bound"])
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
                pixel_values=pixel_values,
                tgt_sizes=tgt_sizes,
                image_bound=image_bound,
                position_ids=data["position_ids"].cuda() if "position_ids" in data else None,
            )
        else:
            # Qwen2.5-VL
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
                pixel_values=data["pixel_values"].cuda(),
                image_grid_thw=data["image_grid_thw"].cuda(),
            )
    else:
        if is_online:
            # we generate the eagle3 using the target model in an online fashion
            eagle3_data = target_model.generate_eagle3_data(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            input_ids = get_dp_data_shard_from_tp(eagle3_data.input_ids)
            attention_mask = get_dp_data_shard_from_tp(eagle3_data.attention_mask)
            loss_mask = get_dp_data_shard_from_tp(eagle3_data.loss_mask)
            target = get_dp_data_shard_from_tp(eagle3_data.target)
            hidden_states = get_dp_data_shard_from_tp(eagle3_data.hidden_states)
        else:
            # we generate the logits using the hidden states loaded from disk
            attention_mask = data["attention_mask"].cuda()
            hidden_states = data["hidden_state"].cuda()
            input_ids, target, loss_mask = target_model.preprocess(
                data["input_ids"], data["target"], data["loss_mask"]
            )
            input_ids = input_ids.cuda()
            target = target_model(
                target.cuda()
            )  # The `data['target']` value occupies a large amount of GPU memory, with a shape of [seqlen, vocab_size]. It needs to be processed before being loaded into the GPU.
            loss_mask = loss_mask.cuda()
        plosses, _, acces = eagle3_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            target=target,
            hidden_states=hidden_states,
        )
    return plosses, acces


def run_backward_and_update(
    args: Namespace, plosses: List[torch.Tensor], optimizer: Optimizer, global_step: int
) -> None:
    ploss_weight = [0.8**i for i in range(len(plosses))]
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


def get_dp_data_shard_from_tp(tensor: torch.Tensor, sp_dim: int = 1) -> torch.Tensor:
    """
    Process: TP split -> Pad to Max Len -> SP gather.
    """
    # 1. TP: Slice the tensor along the batch dimension
    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)

    local_tp_shard = tensor.chunk(tp_size, dim=0)[tp_rank]

    # 2. SP: Handle dynamic sequence lengths and Gather
    sp_group = get_draft_sp_group()

    if sp_group is not None and dist.get_world_size(sp_group) > 1:
        sp_world_size = dist.get_world_size(sp_group)
        local_seq_len = local_tp_shard.size(sp_dim)

        # Find global max sequence length in SP group
        len_tensor = torch.tensor(
            [local_seq_len], device=local_tp_shard.device, dtype=torch.long
        )
        dist.all_reduce(len_tensor, op=dist.ReduceOp.MAX, group=sp_group)
        max_seq_len = len_tensor.item()

        # Pad local tensor if necessary
        # Shape is [Batch, Seq, Hidden] or [Batch, Seq], and sp_dim=1
        if local_seq_len < max_seq_len:
            pad_size = max_seq_len - local_seq_len

            pad_config = [0] * (local_tp_shard.ndim * 2)

            pad_idx = (local_tp_shard.ndim - 1 - sp_dim) * 2 + 1
            pad_config[pad_idx] = pad_size

            # Pad value: 0 is standard, ensure it matches your pad_token_id logic if needed
            local_tp_shard_padded = nn.F.pad(local_tp_shard, pad_config, value=0)
        else:
            local_tp_shard_padded = local_tp_shard

        gathered_shards = [
            torch.empty_like(local_tp_shard_padded) for _ in range(sp_world_size)
        ]
        dist.all_gather(
            gathered_shards, local_tp_shard_padded.contiguous(), group=sp_group
        )

        return torch.cat(gathered_shards, dim=sp_dim)

    return local_tp_shard


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
    draft_model_config, draft_model = build_draft_model(args)
    target_model, processor = build_target_model(args, draft_model_config, is_online)

    # ================================================
    # 3. Build dataloader
    # ================================================
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders(
        args, draft_model_config
    )

    # we load the vocab mapping then
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

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
    target_model_type = getattr(draft_model_config, "target_model_type", None)
    if args.is_vlm and target_model_type == "qwen2_5_vl":
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    elif args.is_vlm and target_model_type == "minicpm_v_4":
        eagle3_model = MiniCPMVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
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

    # ================================================
    # 6. Build tracker
    # ================================================
    tracker = build_tracker(args, parser)
    global_step = 0
    start_epoch = 0
    dist.barrier()

    last_time = time.time()

    # ================================================
    # 7. Start training
    # ================================================
    print_on_rank0(f"Starting training from epoch {start_epoch}")

    # Print first training sample for debugging (use --debug-first-sample to enable)
    if args.debug_first_sample and dist.get_rank() == 0:
        first_sample = next(iter(train_dataloader))
        # Get tokenizer from processor or directly
        _tokenizer = processor.tokenizer if processor is not None else tokenizer
        
        print("\n" + "=" * 60)
        print("First Training Sample")
        print("=" * 60)
        
        # Input tokens
        input_ids = first_sample['input_ids'][0]
        print(f"[INPUT_IDS] shape: {first_sample['input_ids'].shape}")
        print(f"[INPUT_IDS] {input_ids.tolist()}")
        print(f"[INPUT_TEXT] {_tokenizer.decode(input_ids, skip_special_tokens=False)}")
        
        # Label tokens (tokens that contribute to loss)
        mask_indices = (first_sample['loss_mask'][0] == 1).nonzero(as_tuple=True)[0]
        if len(mask_indices) > 0:
            label_ids = input_ids[mask_indices]
            print(f"[LABEL_IDS] count: {len(mask_indices)}, ids: {label_ids.tolist()}")
            print(f"[LABEL_TEXT] {_tokenizer.decode(label_ids, skip_special_tokens=False)}")
        
        print("=" * 60 + "\n")

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

        for data in progress_bar:
            # Skip empty batches (all samples were invalid)
            if data is None:
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
                args, eagle3_model, data, target_model, is_online,
                vlm_type=getattr(draft_model_config, "target_model_type", None),
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
                            args, eagle3_model, data, target_model, is_online,
                            vlm_type=getattr(draft_model_config, "target_model_type", None),
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

        # Save checkpoint at end of epoch if requested
        if args.save_per_epoch:
            print_on_rank0(f"Epoch {epoch} completed, saving checkpoint...")
            save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)

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
