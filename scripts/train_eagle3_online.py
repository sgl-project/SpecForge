"""
This script is used to train a eagle3 model.

"""

import argparse
import dataclasses
import hashlib
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from rich.logging import RichHandler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig

from specforge import BF16Optimizer, SpecForgeArgs, build_tracker, parse_specforge_args
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import (
    destroy_distributed,
    get_draft_dp_group,
    get_draft_dp_rank,
    get_draft_dp_size,
    get_draft_tp_group,
    get_draft_tp_rank,
    get_target_dp_group,
    get_target_dp_size,
    init_distributed,
)
from specforge.modeling.draft import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    Eagle3DraftModel,
    load_param_from_target_model,
)
from specforge.modeling.eagle3 import OnlineEagle3Model, QwenVLOnlineEagle3Model
from specforge.modeling.target import Eagle3TargetOutput, get_eagle3_target_model
from specforge.utils import (
    create_draft_config_from_target,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


class TrainDataLoaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0,
        steps_consumed_in_current_epoch: int = 0,
        steps_consumed: int = 0,
    ):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.steps_consumed_in_current_epoch = steps_consumed_in_current_epoch
        self.steps_consumed = steps_consumed

        self.max_global_steps = len(dataloader) * num_epochs
        self.epoch = start_epoch

    def __iter__(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            self.dataloader.sampler.set_epoch(epoch)
            dataloader = iter(self.dataloader)
            # skip some steps if needed
            if self.steps_consumed_in_current_epoch > 0:
                if self.steps_consumed_in_current_epoch >= len(dataloader):
                    self.steps_consumed_in_current_epoch = 0
                    continue
                print_on_rank0(
                    f"Skipping {self.steps_consumed_in_current_epoch} steps in current epoch"
                )
                # it takes 5 minutes to run 279310 steps
                for _ in tqdm(
                    range(self.steps_consumed_in_current_epoch), desc="Skipping steps"
                ):
                    next(dataloader)
            for data in dataloader:
                self.steps_consumed_in_current_epoch += 1
                self.steps_consumed += 1
                yield data
            self.steps_consumed_in_current_epoch = 0

    def __len__(self):
        return self.max_global_steps - self.steps_consumed

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "steps_consumed_in_current_epoch": self.steps_consumed_in_current_epoch,
            "steps_consumed": self.steps_consumed,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.start_epoch = state_dict["epoch"]
        self.steps_consumed_in_current_epoch = state_dict[
            "steps_consumed_in_current_epoch"
        ]
        self.steps_consumed = state_dict["steps_consumed"]


class Eagle3Trainer:
    def __init__(self, args):
        if not dist.is_initialized():
            init_distributed(
                timeout=args.dist_timeout,
                target_tp_size=args.target_tp_size,
                draft_tp_size=args.draft_tp_size,
            )
        self.trainer_args = SpecForgeArgs.from_cli_args(args)

        set_seed(self.trainer_args.seed)
        self.build_eagle3_model()
        self.train_dataloader, self.eval_dataloader = self.build_dataloaders()
        assert getattr(
            self.eagle3_model.draft_model, "vocab_mapping_loaded", False
        ), "Vocab mapping is not loaded"
        self._auto_calculate_num_steps()
        self.optimizer = self.build_optimizer()
        self.tracker = build_tracker(self.trainer_args, self.trainer_args.output_dir)
        self.micro_batch_idx = 0
        self.global_batch_idx = 0
        if self.trainer_args.draft_model_last_checkpoint is not None:
            if self.trainer_args.enable_zero2:
                draft_tp_group = get_draft_tp_group()
                draft_tp_rank = dist.get_rank(draft_tp_group)
                draft_dp_group = get_draft_dp_group()
                draft_dp_rank = dist.get_rank(draft_dp_group)
                state_path = os.path.join(
                    self.trainer_args.draft_model_last_checkpoint,
                    f"training_state_draft_tp{draft_tp_rank}_dp{draft_dp_rank}.pt",
                )
            else:
                state_path = os.path.join(
                    self.trainer_args.draft_model_last_checkpoint, "training_state.pt"
                )
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                self.global_batch_idx = state.get("global_batch_idx", 0)
                self.micro_batch_idx = state.get("micro_batch_idx", 0)
                self.optimizer.load_state_dict(state)
                self.train_dataloader.load_state_dict(state)
                print_on_rank0(
                    f"Resuming from step {self.train_dataloader.steps_consumed} with {self.global_batch_idx=} and {self.micro_batch_idx=}"
                )
            else:
                print_on_rank0(
                    f"Warning: Checkpoint directory {self.draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
                )

        dist.barrier()

    def _auto_calculate_num_steps(self):
        args = self.trainer_args
        self.steps_per_epoch = math.ceil(
            len(self.train_dataloader.dataloader) / args.draft_accumulation_steps
        )
        print_on_rank0(
            f"Auto-Calculated {self.steps_per_epoch=} = ceil({len(self.train_dataloader.dataloader)=} / {args.draft_accumulation_steps})"
        )

        if args.total_steps is None:
            args.total_steps = args.num_epochs * self.steps_per_epoch
            print_on_rank0(
                f"Auto-Calculated {args.total_steps=} {args.num_epochs=} * {self.steps_per_epoch=}"
            )
        else:
            print_on_rank0(f"Using provided {args.total_steps=}")

        if args.eval_interval == -1:
            args.eval_interval = self.steps_per_epoch
            print_on_rank0(f"Auto-set eval_interval to {self.steps_per_epoch=}")
        if args.save_interval == -1:
            args.save_interval = self.steps_per_epoch
            print_on_rank0(f"Auto-set save_interval to {self.steps_per_epoch=}")

    def _build_target_model(self, draft_model_config: AutoDraftModelConfig):
        args = self.trainer_args
        if (
            args.is_vlm
            and draft_model_config.target_model_type == "qwen2_5_vl"
            and args.target_tp_size == 1
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
            target_model = get_eagle3_target_model(
                pretrained_model_name_or_path=args.target_model_path,
                backend=args.target_model_backend,
                torch_dtype=torch.bfloat16,
                device="cuda",
                cache_dir=args.cache_dir,
            )

        if (
            hasattr(self.draft_model_config, "eagle_config")
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
        print_with_rank("...Initialized target model")
        return target_model, processor

    def _build_draft_model(
        self, param_dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[PretrainedConfig, Eagle3DraftModel]:
        args = self.trainer_args
        if args.draft_model_config is None:
            # Auto-generate and save config file
            auto_config_path = create_draft_config_from_target(
                target_model_path=args.target_model_path, cache_dir=args.cache_dir
            )
            draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
        else:
            # Use provided config file
            draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)

        if args.draft_model_last_checkpoint:
            draft_model: Eagle3DraftModel = AutoEagle3DraftModel.from_pretrained(
                args.draft_model_last_checkpoint,
                attention_backend=args.draft_attention_backend,
                torch_dtype=param_dtype,
            ).cuda()
        else:
            draft_model: Eagle3DraftModel = AutoEagle3DraftModel.from_config(
                draft_model_config,
                attention_backend=args.draft_attention_backend,
                torch_dtype=param_dtype,
            ).cuda()
            # draft_model.sync_state_dict_across_tp()
        load_param_from_target_model(
            target_model_path=args.target_model_path,
            param_key=args.embedding_key,
            param=draft_model.get_embedding_param(),
        )
        draft_model.freeze_embedding()
        print_with_rank("...Initialized draft model")
        return draft_model_config, draft_model

    def build_eagle3_model(self):
        param_dtype = torch.bfloat16
        args = self.trainer_args
        self.draft_model_config, self.draft_model = self._build_draft_model(param_dtype)
        self.target_model, self.target_processor = self._build_target_model(
            self.draft_model_config
        )
        if (
            args.is_vlm
            and getattr(self.draft_model_config, "target_model_type", None)
            == "qwen2_5_vl"
        ):
            eagle3_model = QwenVLOnlineEagle3Model(
                target_model=self.target_model,
                draft_model=self.draft_model,
                processor=self.target_processor,
                length=args.ttt_length,
                attention_backend=args.draft_attention_backend,
            )
        # elif args.target_model_backend == "sglang":
        #     assert isinstance(
        #         self.target_model, SGLangEagle3TargetModel
        #     ), "Target model must be a SGLangEagle3TargetModel"
        #     eagle3_model = OfflineEagle3Model(
        #         draft_model=self.draft_model,
        #         length=args.ttt_length,
        #         attention_backend=args.draft_attention_backend,
        #     )
        #     # load_param_from_target_model(
        #     #     target_model_path=args.target_model_path,
        #     #     param_key=args.lm_head_key,
        #     #     param=eagle3_model.target_head.fc.weight,
        #     # )
        #     # eagle3_model.target_head.freeze_parameters()
        else:
            eagle3_model = OnlineEagle3Model(
                draft_model=self.draft_model,
                length=args.ttt_length,
                attention_backend=args.draft_attention_backend,
            )

        eagle3_model = FSDP(
            eagle3_model,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                buffer_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            sharding_strategy=(
                ShardingStrategy.SHARD_GRAD_OP
                if args.enable_zero2
                else ShardingStrategy.NO_SHARD
            ),
            process_group=get_draft_dp_group(),
        )
        print_with_rank("Initialized Eagle3 FSDP model")
        self.eagle3_model = eagle3_model

    def build_dataloaders(self):
        args = self.trainer_args
        tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
        cache_params_string = (
            f"{args.train_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}"  # Tokenizer may also different
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
        train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
        target_dp_size = get_target_dp_size()
        draft_dp_size = get_draft_dp_size()
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
                processor=self.target_processor,
                num_proc=args.build_dataset_num_proc,
            )
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=self.draft_model_config.vocab_size,
                draft_vocab_size=self.draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )

        train_dataloader = prepare_dp_dataloaders(
            train_eagle3_dataset,
            args.draft_micro_batch_size * max(draft_dp_size // target_dp_size, 1),
            num_workers=4,
            shuffle=True,
            prefetch_factor=8,
            process_group=(
                get_target_dp_group()
                if target_dp_size < draft_dp_size
                else get_draft_dp_group()
            ),
            is_vlm=args.is_vlm,
        )
        print_with_rank(f"...Initialized train dataloader")

        # we load the vocab mapping then
        self.draft_model.load_vocab_mapping(vocab_mapping_path)
        print_with_rank(f"...Loaded vocab mapping")

        eval_dataloader = None
        if args.eval_data_path is not None:
            cache_params_string = (
                f"{args.eval_data_path}-"
                f"{args.max_length}-"
                f"{args.chat_template}-"
                f"{args.target_model_path}"  # Tokenizer may also different
            )
            test_cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=test_cache_key,
                processor=self.target_processor,
                num_proc=args.build_dataset_num_proc,
                is_vlm=args.is_vlm,
                is_preformatted=args.is_preformatted,
            )
            eval_dataloader = prepare_dp_dataloaders(
                eval_eagle3_dataset,
                max(draft_dp_size // target_dp_size, 1),
                num_workers=4,
                shuffle=False,
                prefetch_factor=8,
                process_group=(
                    get_target_dp_group()
                    if target_dp_size < draft_dp_size
                    else get_draft_dp_group()
                ),
                is_vlm=args.is_vlm,
            )
            print_with_rank(f"...Initialized eval dataloader")

        return (
            TrainDataLoaderWrapper(train_dataloader, args.num_epochs),
            eval_dataloader,
        )

    def build_optimizer(self):
        args = self.trainer_args
        optimizer = BF16Optimizer(
            self.eagle3_model,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=args.total_steps,
            enable_zero2=args.enable_zero2,
        )
        return optimizer

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_checkpoint(self, step: int):
        # Save the model
        print_on_rank0(f"save_checkpoint on {step=}")
        args = self.trainer_args
        if args.save_per_epoch:
            epoch_output_dir = os.path.join(
                args.output_dir, f"epoch_{self.train_dataloader.epoch}"
            )
        else:
            epoch_output_dir = os.path.join(args.output_dir, f"step_{step}")

        if dist.get_rank() == 0:
            os.makedirs(epoch_output_dir, exist_ok=True)
        dist.barrier()

        with FSDP.state_dict_type(self.eagle3_model, StateDictType.FULL_STATE_DICT):
            model_state_dict = self.eagle3_model.state_dict()
            state_to_save = {
                "args": self.trainer_args,
                "global_batch_idx": self.global_batch_idx,
                "micro_batch_idx": self.micro_batch_idx,
            }
            state_to_save.update(self.train_dataloader.state_dict())
            state_to_save.update(self.optimizer.state_dict())
            draft_model_state_dict = {
                k.replace("draft_model.", ""): v
                for k, v in model_state_dict.items()
                if "draft_model." in k and "embed" not in k.lower()
            }

            draft_dp_rank = get_draft_dp_rank()
            draft_tp_rank = get_draft_tp_rank()
            if not args.enable_zero2:
                training_state_path = os.path.join(
                    epoch_output_dir, f"training_state_draft_tp{draft_tp_rank}.pt"
                )
                if draft_dp_rank == 0:
                    torch.save(state_to_save, training_state_path)
                    print_with_rank(
                        f"Saved full training state to {training_state_path}"
                    )
            else:
                training_state_path = os.path.join(
                    epoch_output_dir,
                    f"training_state_draft_tp{draft_tp_rank}_dp{draft_dp_rank}.pt",
                )
                torch.save(state_to_save, training_state_path)
                print_with_rank(f"Saved full training state to {training_state_path}")

            if draft_dp_rank == 0:
                self.draft_model.save_pretrained(
                    epoch_output_dir,
                    state_dict=draft_model_state_dict,
                )
                print_with_rank(f"Saved draft model weights to {epoch_output_dir}")

            if dist.get_rank() == 0:
                if args.max_num_saved_checkpoints > 1:
                    if args.save_per_epoch:
                        step_re = re.compile(r"^epoch_(\d+)$")
                    else:
                        step_re = re.compile(r"^step_(\d+)$")
                    dirs = []
                    for name in os.listdir(args.output_dir):
                        full_path = os.path.join(args.output_dir, name)
                        if not os.path.isdir(full_path):
                            continue
                        match = step_re.match(name)
                        if match:
                            dirs.append((match.group(1), full_path))
                        else:
                            print_with_rank(
                                f"Warning: {name} is not a valid step directory"
                            )
                    dirs.sort(key=lambda x: int(x[0]), reverse=True)
                    for i, (_, path) in enumerate(dirs):
                        if i >= args.max_num_saved_checkpoints:
                            print_with_rank(f"Removing {path}")
                            shutil.rmtree(path)

            dist.barrier()

    def record_metrics(self, logdict: Dict[str, torch.Tensor], mode: str = "train"):
        if mode == "eval":
            for key, value in logdict.items():
                dist.all_reduce(value, op=dist.ReduceOp.AVG)
                logdict[key] = value.item()
        self.tracker.log(logdict, step=self.global_batch_idx)

    def _generate_eagle3_data(
        self, data: Dict[str, torch.Tensor]
    ) -> List[Eagle3TargetOutput]:
        args = self.trainer_args
        if args.is_vlm:
            return [data]
        return self.target_model.generate_eagle3_data(
            input_ids=data["input_ids"].cuda(),
            attention_mask=data["attention_mask"].cuda(),
            loss_mask=data["loss_mask"].cuda(),
        )

    def _forward_step(self, data: Eagle3TargetOutput):
        args = self.trainer_args
        if args.is_vlm:
            plosses, _, acces = self.eagle3_model(
                input_ids=data["input_ids"].cuda(),  # [B, S]
                attention_mask=data["attention_mask"].cuda(),  # [B, S]
                loss_mask=data[
                    "loss_mask"
                ].cuda(),  # [B, S] This is different from the online version
                pixel_values=data["pixel_values"].cuda(),
                image_grid_thw=data["image_grid_thw"].cuda(),
            )
        else:
            plosses, _, acces = self.eagle3_model(
                input_ids=data.input_ids,
                attention_mask=data.attention_mask,
                loss_mask=data.loss_mask,
                target=data.target,
                hidden_states=data.hidden_states,
            )
        return plosses, acces

    def _backward_step(self, plosses: List[torch.Tensor], acces: List[float]):
        args = self.trainer_args
        self.micro_batch_idx += 1
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = (
            sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            / args.draft_accumulation_steps
        )
        ploss.backward()
        do_optimizer_step = self.micro_batch_idx % args.draft_accumulation_steps == 0
        if do_optimizer_step:
            self.global_batch_idx += 1
            self.optimizer.step()
        return do_optimizer_step

    @torch.no_grad()
    def eval(self, logdict: Optional[Dict[str, Union[torch.Tensor, float]]] = None):
        if self.eval_dataloader is None:
            return
        self.eagle3_model.eval()
        args = self.trainer_args
        # Run evaluation
        eval_acces = [[] for _ in range(args.ttt_length)]
        eval_plosses = [[] for _ in range(args.ttt_length)]

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                self.eval_dataloader,
                desc=f"Evaluating global_batch_idx={self.global_batch_idx}, epoch={self.train_dataloader.epoch}",
                leave=True,
            )
        else:
            progress_bar = self.eval_dataloader
        with torch.no_grad():
            for data in progress_bar:
                eagle3_data = self._generate_eagle3_data(data)
                plosses, acces = self._forward_step(eagle3_data)
                for i in range(len(eval_plosses)):
                    eval_plosses[i].append(plosses[i].item())
                for i in range(len(eval_acces)):
                    eval_acces[i].append(acces[i])

        if logdict is None:
            logdict = {}
        for i in range(len(eval_plosses)):
            logdict[f"eval/epochploss_{i}"] = (
                torch.tensor(eval_plosses[i]).cuda().mean()
            )
        for i in range(len(eval_acces)):
            logdict[f"eval/epochacc_{i}"] = torch.tensor(eval_acces[i]).cuda().mean()
        self.record_metrics(logdict, mode="eval")

    def train(self):
        args = self.trainer_args
        train_plosses = [[] for _ in range(args.ttt_length)]
        train_acces = [[] for _ in range(args.ttt_length)]
        self.train_logdict = defaultdict(float)
        self.draft_model.train()
        pbar = None
        if dist.get_rank() == 0:
            pbar = async_tqdm(
                total=min(args.total_steps, self.steps_per_epoch * args.num_epochs),
                initial=self.global_batch_idx,
                desc="Training...",
                leave=True,
                dynamic_ncols=True,
            )
        for data in self.train_dataloader:
            eagle3_data = self._generate_eagle3_data(data)
            plosses, acces = self._forward_step(eagle3_data)
            do_optimizer_step = self._backward_step(plosses, acces)
            for i in range(len(train_plosses)):
                train_plosses[i].append(plosses[i].item())
                self.train_logdict[f"train/ploss_{i}"] += (
                    plosses[i].item() / args.draft_accumulation_steps
                )
            for i in range(len(train_acces)):
                train_acces[i].append(acces[i])
                self.train_logdict[f"train/acc_{i}"] += (
                    acces[i] / args.draft_accumulation_steps
                )

            if do_optimizer_step:
                if pbar is not None:
                    pbar.update(1)
                if self.global_batch_idx % args.log_interval == 0:
                    self.train_logdict["train/lr"] = self.optimizer.get_learning_rate()
                    self.record_metrics(self.train_logdict, mode="train")
                    self.train_logdict = defaultdict(float)

                if self.global_batch_idx % args.save_interval == 0:
                    self.save_checkpoint(self.global_batch_idx)

                if self.global_batch_idx % args.eval_interval == 0:
                    logdict = {}
                    for i in range(len(train_acces)):
                        logdict[f"train/epochacc_{i}"] = (
                            torch.tensor(train_acces[i]).cuda().mean()
                        )
                    for i in range(len(train_plosses)):
                        logdict[f"train/epochploss_{i}"] = (
                            torch.tensor(train_plosses[i]).cuda().mean()
                        )
                    self.eval(logdict)
                    train_plosses = [[] for _ in range(args.ttt_length)]
                    train_acces = [[] for _ in range(args.ttt_length)]

                if self.global_batch_idx >= args.total_steps:
                    break
        destroy_distributed()
        return


def main():
    # initialize
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler()],
        force=True,
    )
    args = parse_specforge_args()
    trainer = Eagle3Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
