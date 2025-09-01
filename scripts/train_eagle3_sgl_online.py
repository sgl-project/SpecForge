import argparse
import hashlib
import math
import os
import re
import shutil
from collections import defaultdict

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from datasets import load_dataset
from sglang.bench_one_batch import BenchArgs
from sglang.srt.server_args import ServerArgs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OfflineEagle3Model
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.data.utils import DataCollatorWithPadding
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.target.sgl_model_wrapper import SglangTargetModel
from specforge.modeling.target.target_head import TargetHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import NoOpTracker, create_tracker, get_tracker_class
from specforge.utils import (
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
    rank_0_priority,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument(
        "--lm-head-key",
        type=str,
        default="lm_head.weight",
        help="The key of the lm head weight to load from the target model",
    )

    # add training-related arguments
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    parser.add_argument("--draft-global-batch-size", type=int, default=16)
    parser.add_argument("--draft-micro-batch-size", type=int, default=1)
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )

    # # data processing type
    # "--dist-timeout", "--chat-template", "--tp-size", "--batch-size" is includued in sgl server_args

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--log-interval", type=int, default=-1)
    parser.add_argument("--eval-interval", type=int, default=-1)
    parser.add_argument("--save-interval", type=int, default=-1)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps. If not provided, will be calculated as num_epochs * steps_per_epoch",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=-1,
        help="The total number of checkpoints to save. If -1, save all checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--draft-attention-backend",
        type=str,
        default="flex_attention",
        choices=["flex_attention", "sdpa"],
    )

    # resume
    parser.add_argument("--resume", action="store_true")

    # report backend
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
        help="The integration to report results and logs to.",
    )
    # wandb-specific args
    parser.add_argument(
        "--wandb-project", type=str, default=None, help="The project name for W&B."
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="The run name for W&B."
    )
    parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
    # add swanlab-specific args ---
    parser.add_argument(
        "--swanlab-project",
        type=str,
        default=None,
        help="The project name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-name",
        type=str,
        default=None,
        help="The experiment name for swanlab.",
    )
    parser.add_argument(
        "--swanlab-key",
        type=str,
        default=None,
        help="The API key for swanlab non-interactive login.",
    )
    # mlflow-specific args
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="The MLflow run name. If not set, MLflow will auto-generate one.",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile-start-step", type=int, default=30)
    parser.add_argument("--profile-num-steps", type=int, default=4)

    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)

    args = parser.parse_args()
    return parser, args


from torch.utils.data import DataLoader


class TrainDataLoaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0,
        steps_consumed_in_current_epoch: int = 0,
        steps_consumed: int = 0,
        max_global_steps: int = None,
    ):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.steps_consumed_in_current_epoch = steps_consumed_in_current_epoch
        self.steps_consumed = steps_consumed

        self.max_global_steps = len(dataloader) * num_epochs
        if max_global_steps is not None:
            self.max_global_steps = min(self.max_global_steps, max_global_steps)
        self.epoch = start_epoch

    def __iter__(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            self.dataloader.sampler.set_epoch(epoch)
            dataloader = iter(self.dataloader)
            # skip some steps if needed
            for _ in range(self.steps_consumed_in_current_epoch):
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


class SglOnlineEagle3Trainer:
    def __init__(self, args):
        # using sglang server args and renaming is needed
        args.tp_size = args.tensor_parallel_size
        args.target_batch_size = args.tp_size * 8
        set_seed(args.seed)
        self.args = args
        if not dist.is_initialized():
            init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
        args.dp_size = dist.get_world_size()  #  // args.tp_size
        args.draft_accumulation_steps = (
            args.draft_global_batch_size // args.dp_size // args.draft_micro_batch_size
        )
        assert (
            args.draft_accumulation_steps * args.draft_micro_batch_size * args.dp_size
            == args.draft_global_batch_size
        ), f"draft_global_batch_size={args.draft_global_batch_size} must be divisible by dp_size={args.dp_size} and micro_batch_size={args.draft_micro_batch_size}"
        print_with_rank(
            f"draft_accumulation_steps={args.draft_global_batch_size} // {args.dp_size} // {args.draft_micro_batch_size}={args.draft_accumulation_steps}"
        )
        assert (
            args.draft_micro_batch_size == 1
        ), "draft_micro_batch_size must be 1 for SglOnlineEagle3Trainer"
        assert (
            args.tp_size == dist.get_world_size()
        ), "tp_size must be equal to world_size for SglOnlineEagle3Trainer"
        self.draft_model_config = AutoDraftModelConfig.from_file(
            self.args.draft_model_config
        )
        self.draft_model_last_checkpoint = None
        if args.resume and os.path.isdir(args.output_dir):
            print_on_rank0(args.output_dir)
            self.draft_model_last_checkpoint = get_last_checkpoint(
                args.output_dir, prefix="step"
            )
            print_on_rank0(
                f"Last checkpoint detected: {self.draft_model_last_checkpoint}"
            )

        self.create_eagle3_model()
        self.train_dataloader, self.eval_dataloader = self.create_dataloaders()
        assert getattr(
            self.eagle3_model.draft_model, "vocab_mapping_loaded", False
        ), "Vocab mapping is not loaded"
        self.shard_model()
        if args.total_steps is None:
            steps_per_epoch = math.ceil(
                len(self.train_dataloader) / args.draft_accumulation_steps
            )
            args.total_steps = args.num_epochs * steps_per_epoch
            print_on_rank0(
                f"Auto-Calculated {args.total_steps=}={args.num_epochs=} * {steps_per_epoch=}"
            )
        else:
            print_on_rank0(f"Using provided {args.total_steps=}")
        self.optimizer = self.create_optimizer()

        self.tracker = NoOpTracker(self.args, self.args.output_dir)
        if self.draft_model_last_checkpoint is not None:
            print_on_rank0(
                f"Resuming draft model from {self.draft_model_last_checkpoint}"
            )
            state_path = os.path.join(
                self.draft_model_last_checkpoint, "training_state.pt"
            )
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                self.optimizer.load_state_dict(state)
                self.train_dataloader.load_state_dict(state)
                print_on_rank0(
                    f"Resuming from step {self.train_dataloader.steps_consumed}"
                )
            else:
                print_on_rank0(
                    f"Warning: Checkpoint directory {self.draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
                )

        dist.barrier()
        self.step_idx = 0

    def _create_target_model(self):
        target_model = SglangTargetModel(
            args=self.args,
            target_micro_batch_size=self.args.tp_size,
            draft_micro_batch_size=self.args.draft_micro_batch_size,
            tp_group=dist.group.WORLD,
            enable_aux_hidden_states=True,
            return_full_logits=False,
        )
        if (
            hasattr(self.draft_model_config, "eagle_config")
            and self.draft_model_config.eagle_config is not None
            and "eagle_aux_hidden_state_layer_ids"
            in self.draft_model_config.eagle_config
        ):
            target_model.set_aux_hidden_states_layers(
                self.draft_model_config.eagle_config["eagle_aux_hidden_state_layer_ids"]
            )
        else:
            target_model.set_aux_hidden_states_layers()
        self.target_model = target_model

    def _create_draft_model(self, param_dtype=torch.bfloat16):
        if self.draft_model_last_checkpoint:
            draft_model = (
                AutoEagle3DraftModel.from_pretrained(
                    self.draft_model_last_checkpoint,
                    attention_backend=self.args.draft_attention_backend,
                )
                .cuda()
                .to(param_dtype)
            )
        else:
            draft_model = (
                AutoEagle3DraftModel.from_config(
                    self.draft_model_config,
                    attention_backend=self.args.draft_attention_backend,
                )
                .cuda()
                .to(param_dtype)
            )
        draft_model.load_embedding(
            self.args.target_model_path, embedding_key=self.args.embedding_key
        )
        draft_model.freeze_embedding()
        self.draft_model = draft_model

    def create_eagle3_model(self):
        param_dtype = torch.bfloat16
        args = self.args

        target_head = TargetHead(args.target_model_path)
        target_head.load_weights(
            model_path=args.target_model_path, lm_head_key=args.lm_head_key
        )
        target_head.freeze_weights()
        target_head = target_head.eval().cuda().to(param_dtype)
        self._create_target_model()
        self._create_draft_model(param_dtype)
        eagle3_model = OfflineEagle3Model(
            target_model=self.target_model,
            target_head=target_head,
            draft_model=self.draft_model,
            length=self.args.ttt_length,
            attention_backend=self.args.draft_attention_backend,
        )
        self.eagle3_model = eagle3_model

    def create_dataloaders(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_model_path)
        cache_params_string = (
            f"{self.args.train_data_path}-"
            f"{self.args.max_length}-"
            f"{self.args.chat_template}-"
            f"{self.args.target_model_path}"  # Tokenizer may also different
        )
        cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
        train_dataset = load_dataset("json", data_files=self.args.train_data_path)[
            "train"
        ]
        with rank_0_priority():
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=self.args.chat_template,
                max_length=self.args.max_length,
                cache_dir=os.path.join(self.args.cache_dir, "processed_dataset"),
                cache_key=cache_key,
            )
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=self.draft_model_config.vocab_size,
                draft_vocab_size=self.draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(self.args.cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )
        train_dataloader = prepare_dp_dataloaders(
            train_eagle3_dataset,
            1,
            num_workers=4,
            shuffle=True,
            process_group=get_dp_group(),
        )
        print_with_rank(f"Initialized train dataloader")

        # we load the vocab mapping then
        self.draft_model.load_vocab_mapping(vocab_mapping_path)
        print_with_rank(f"Loaded vocab mapping")

        eval_dataloader = None
        if self.args.eval_data_path is not None:
            cache_params_string = (
                f"{self.args.eval_data_path}-"
                f"{self.args.max_length}-"
                f"{self.args.chat_template}-"
                f"{self.args.target_model_path}"  # Tokenizer may also different
            )
            test_cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
            eval_dataset = load_dataset("json", data_files=self.args.eval_data_path)[
                "train"
            ]
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                self.args.chat_template,
                self.args.max_length,
                cache_dir=os.path.join(self.args.cache_dir, "processed_dataset"),
                cache_key=test_cache_key,
            )
            eval_dataloader = prepare_dp_dataloaders(
                eval_eagle3_dataset,
                1,
                num_workers=4,
                shuffle=False,
                process_group=get_dp_group(),
            )
            print_with_rank(f"Initialized eval dataloader")

        if self.args.eval_interval == -1:
            self.args.eval_interval = (
                len(train_dataloader)
                // self.args.target_batch_size
                * self.args.target_batch_size
            )
            print_on_rank0(f"Auto-set eval_interval to {self.args.eval_interval}")
        if self.args.save_interval == -1:
            self.args.save_interval = (
                len(train_dataloader)
                // self.args.target_batch_size
                * self.args.target_batch_size
            )
            print_on_rank0(f"Auto-set save_interval to {self.args.save_interval}")
        return (
            TrainDataLoaderWrapper(train_dataloader, self.args.num_epochs),
            eval_dataloader,
        )

    def shard_model(self):
        # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
        self.eagle3_model = FSDP(
            self.eagle3_model,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                buffer_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            sharding_strategy=ShardingStrategy.NO_SHARD,
            ignored_modules=[],
            process_group=dist.group.WORLD,  # get_dp_group(),
        )

    def create_optimizer(self):
        optimizer = BF16Optimizer(
            self.eagle3_model,
            lr=self.args.learning_rate,
            max_grad_norm=self.args.max_grad_norm,
            warmup_ratio=self.args.warmup_ratio,
            total_steps=self.args.total_steps,
        )
        return optimizer

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_checkpoint(self, step: int):
        # Save the model
        epoch_output_dir = os.path.join(self.args.output_dir, f"step_{step}")

        if dist.get_rank() == 0:
            os.makedirs(epoch_output_dir, exist_ok=True)
        dist.barrier()

        with FSDP.state_dict_type(self.eagle3_model, StateDictType.FULL_STATE_DICT):
            model_state_dict = self.eagle3_model.state_dict()
            state_to_save = {
                "args": self.args,
            }
            state_to_save.update(self.train_dataloader.state_dict())
            state_to_save.update(self.optimizer.state_dict())
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
                self.draft_model.save_pretrained(
                    epoch_output_dir,
                    state_dict=draft_model_state_dict,
                )
                print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                if self.args.save_total_limit > 1:
                    step_re = re.compile(r"^step_(\d+)$")
                    dirs = []
                    for name in os.listdir(self.args.output_dir):
                        full_path = os.path.join(self.args.output_dir, name)
                        if not os.path.isdir(full_path):
                            continue
                        dirs.append((step_re.match(name).group(1), full_path))
                    dirs.sort(key=lambda x: x[0], reverse=True)
                    for i, (_, path) in enumerate(dirs):
                        if i >= self.args.save_total_limit:
                            print(f"Removing {path}")
                            shutil.rmtree(path)

            dist.barrier()

    def train_step(self, data):
        self.step_idx += 1
        plosses, _, acces = self.eagle3_model(
            input_ids=data["input_ids"].cuda(),  # [B, S]
            attention_mask=data["attention_mask"].cuda(),  # [B, S]
            loss_mask=data["loss_mask"]
            .unsqueeze(-1)
            .cuda(),  # [B, S, 1] This is different from the online version
            hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
            target=data["target"].cuda(),  # [B, S, D*3]
        )

        # calculate weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = (
            sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            / self.args.draft_accumulation_steps
        )
        ploss.backward()
        if self.step_idx % self.args.draft_accumulation_steps == 0:
            self.optimizer.step()
        return plosses, acces

    @torch.no_grad()
    def eval(self):
        if self.eval_dataloader is None:
            return
        draft_data_collator = DataCollatorWithPadding()
        self.eagle3_model.eval()
        # Run evaluation
        eval_acces = [[] for _ in range(self.eagle3_model.module.length)]
        eval_plosses = [[] for _ in range(self.eagle3_model.module.length)]

        data_for_target = []
        for data in tqdm(self.eval_dataloader, desc=f"Evaluating Step {self.step_idx}"):
            data_for_target.append(data)
            if len(data_for_target) >= self.args.target_batch_size:
                torch.cuda.empty_cache()
                data_for_draft = self.target_model.forward(
                    data_for_target, draft_data_collator=draft_data_collator
                )
                torch.cuda.empty_cache()
                data_for_target = []
                for data in data_for_draft:
                    step_plosses, _, step_acces = self.eagle3_model(
                        input_ids=data["input_ids"].cuda(),  # [B, S]
                        attention_mask=data["attention_mask"].cuda(),  # [B, S]
                        loss_mask=data["loss_mask"]
                        .unsqueeze(-1)
                        .cuda(),  # [B, S, 1] This is different from the online version
                        hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
                        target=data["target"].cuda(),  # [B, S, D*3]
                    )
                    for i in range(len(eval_plosses)):
                        eval_plosses[i].append(step_plosses[i].item())
                    for i in range(len(eval_acces)):
                        eval_acces[i].append(step_acces[i])

        torch.cuda.empty_cache()
        data_for_draft = self.target_model.forward(
            data_for_target, draft_data_collator=draft_data_collator
        )
        torch.cuda.empty_cache()
        for data in data_for_draft:
            step_plosses, _, step_acces = self.eagle3_model(
                input_ids=data["input_ids"].cuda(),  # [B, S]
                attention_mask=data["attention_mask"].cuda(),  # [B, S]
                loss_mask=data["loss_mask"]
                .unsqueeze(-1)
                .cuda(),  # [B, S, 1] This is different from the online version
                hidden_states=data["hidden_state"].cuda(),  # [B, S, D]
                target=data["target"].cuda(),  # [B, S, D*3]
            )
            for i in range(len(eval_plosses)):
                eval_plosses[i].append(step_plosses[i].item())
            for i in range(len(eval_acces)):
                eval_acces[i].append(step_acces[i])

        eval_logdict = {}
        for i in range(len(eval_acces)):
            acc_i = torch.tensor(eval_acces[i]).cuda().mean()
            dist.all_reduce(acc_i, op=dist.ReduceOp.AVG)
            eval_logdict[f"eval/epochacc_{i}"] = acc_i.item()

        for i in range(len(eval_plosses)):
            loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i, op=dist.ReduceOp.AVG)
            eval_logdict[f"eval/epochploss_{i}"] = loss_i.item()
        self.tracker.log(eval_logdict, step=self.step_idx)

    def train(self):
        draft_data_collator = DataCollatorWithPadding()
        train_acces = [[] for _ in range(self.eagle3_model.module.length)]
        train_plosses = [[] for _ in range(self.eagle3_model.module.length)]
        self.train_logdict = defaultdict(float)

        data_for_target = []
        self.draft_model.train()
        for _, data in tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Training",
        ):
            data_for_target.append(data)
            if len(data_for_target) >= self.args.target_batch_size:
                torch.cuda.empty_cache()
                data_for_draft = self.target_model.forward(
                    data_for_target, draft_data_collator=draft_data_collator
                )
                data_for_target = []
                for data_ in data_for_draft:
                    step_plosses, step_acces = self.train_step(data_)
                    for i in range(len(train_acces)):
                        train_acces[i].append(step_acces[i])
                        self.train_logdict[f"train/acc_{i}"] += (
                            step_acces[i] / self.args.draft_accumulation_steps
                        )
                    for i in range(len(train_plosses)):
                        train_plosses[i].append(step_plosses[i].item())
                        self.train_logdict[f"train/ploss_{i}"] += (
                            step_plosses[i].item() / self.args.draft_accumulation_steps
                        )
                    if (self.step_idx % self.args.draft_accumulation_steps == 0) and (
                        (self.step_idx // self.args.draft_accumulation_steps)
                        % self.args.log_interval
                        == 0
                    ):
                        self.train_logdict["train/lr"] = (
                            self.optimizer.get_learning_rate()
                        )
                        self.tracker.log(
                            self.train_logdict,
                            step=self.step_idx // self.args.draft_accumulation_steps,
                        )
                        self.train_logdict = defaultdict(float)
                if self.step_idx % self.args.eval_interval == 0:
                    train_logdict = {}
                    for i in range(len(train_acces)):
                        acc_i = torch.tensor(train_acces[i]).cuda().mean()
                        dist.all_reduce(acc_i, op=dist.ReduceOp.AVG)
                        train_logdict[f"train/epochacc_{i}"] = acc_i.item()

                    for i in range(len(train_plosses)):
                        loss_i = torch.tensor(train_plosses[i]).cuda().mean()
                        dist.all_reduce(loss_i, op=dist.ReduceOp.AVG)
                        train_logdict[f"train/epochploss_{i}"] = loss_i.item()
                    self.tracker.log(train_logdict, step=self.step_idx)
                    self.eval()

                if self.step_idx % self.args.save_interval == 0:
                    self.save_checkpoint(self.step_idx)

        destroy_distributed()
        return


def main():
    # initialize
    parser, args = parse_args()
    trainer = SglOnlineEagle3Trainer(args)
    tracker_class = get_tracker_class(args.report_to)
    if tracker_class:
        tracker_class.validate_args(parser, args)
    else:
        parser.error(f"Unknown tracker: {args.report_to}")
    tracker = create_tracker(args, args.output_dir)
    trainer.set_tracker(tracker)
    trainer.train()


if __name__ == "__main__":
    main()
