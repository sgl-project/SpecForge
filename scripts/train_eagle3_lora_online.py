import argparse
import hashlib
import os

import torch
import torch.distributed as dist
import wandb
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# 添加PEFT库导入
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig
import json

from specforge import (
    AutoDistributedTargetModel,
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    OnlineEagle3Model,
)
from specforge.data import (
    build_eagle3_dataset,
    generate_vocab_mapping_file,
    prepare_dp_dataloaders,
)
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import get_last_checkpoint, print_with_rank, rank_0_priority


def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")

    # add model-related arguments
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    
    # --- MODIFICATION START ---
    # 新增参数，用于加载预训练的draft model
    parser.add_argument("--base-draft-model-path", type=str, default=None, help="Path to a pre-trained base draft model")
    # --- MODIFICATION END ---
    
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )

    # LoRA配置
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA training")
    parser.add_argument("--lora-config", type=str, default=None, help="Path to LoRA config file for draft model")
    parser.add_argument("--target-lora-path", type=str, default=None, help="Path to pre-trained target LoRA adapter")

    # add training-related arguments
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)

    # data processing type
    parser.add_argument("--chat-template", type=str, default="llama3")

    # distributed training
    parser.add_argument("--tp-size", type=int, default=1)

    # other args
    parser.add_argument("--cache-key", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=20,
        help="Timeout for collective communication in minutes",
    )
    parser.add_argument(
        "--skip-vocab-mapping",
        action="store_true",
        help="Use pretrained vocab mapping without regeneration",
    )

    # resume
    parser.add_argument("--resume", action="store_true")

    # wandb wandb args
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)

    args = parser.parse_args()
    return args


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_project, name=args.wandb_name)


def wandb_log_if_initialized(log_dict):
    if dist.get_rank() == 0 and wandb.run is not None:
        wandb.log(log_dict)


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def print_trainable_parameters(model, model_name="Model"):
    """打印模型的可训练参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print_with_rank(
        f"{model_name}: {trainable_params:,} trainable parameters out of {all_param:,} total parameters "
        f"({100 * trainable_params / all_param:.2f}% trainable)"
    )


def load_lora_config(lora_config_path, is_trainable=False):
    """从配置文件加载LoRA配置"""
    if not lora_config_path or not os.path.exists(lora_config_path):
        raise ValueError(f"LoRA config file not found: {lora_config_path}")
    
    print_with_rank(f"Loading LoRA config from: {lora_config_path}")
    
    # 从配置文件创建LoraConfig
    with open(lora_config_path, 'r') as f:
        config_dict = json.load(f)

    if is_trainable:
        config_dict["inference_mode"] = False
    
    lora_config = LoraConfig(**config_dict)
    print_with_rank(f"Loaded LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
    
    return lora_config


def main():
    # initialize
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank(f"Initialized distributed environment")

    if args.wandb and dist.get_rank() == 0:
        init_wandb(args)

    # detecting last ckpt for draft model
    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    # build target and draft model
    if args.tp_size > 1:
        # to avoid CPU RAM OOM, we directly init the model on CUDA
        target_model = AutoDistributedTargetModel.from_pretrained(
            pretrained_model_name_or_path=args.target_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
    else:
        target_model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.target_model_path,
                torch_dtype=torch.bfloat16,
            )
            .cuda()
        )
    
    # add target LoRA
    if args.use_lora:
        if args.target_lora_path and os.path.exists(args.target_lora_path):
            print_with_rank(f"Loading pre-trained target LoRA from: {args.target_lora_path}")
            # 从target LoRA路径加载配置
            target_lora_config = load_lora_config(os.path.join(args.target_lora_path, "adapter_config.json"))
            target_model = get_peft_model(target_model, target_lora_config)
            target_model.load_adapter(args.target_lora_path, "default")
            print_with_rank(f"Loaded pre-trained target LoRA adapter")
        else:
            print_with_rank(f"No pre-trained target LoRA specified, using base target model")
        
        # 冻结target模型的所有参数（包括LoRA）
        for param in target_model.parameters():
            param.requires_grad = False
        target_model = target_model.eval()
        print_with_rank(f"Target model frozen for inference")
    else:
        target_model = target_model.eval()
    
    print_with_rank(f"Initialized target model")
    
    # 修改draft model加载逻辑
    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    
    if args.base_draft_model_path and os.path.exists(args.base_draft_model_path):
        # 优先从预训练的draft model路径加载
        print_with_rank(f"Loading base draft model from: {args.base_draft_model_path}")
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(args.base_draft_model_path)
            .cuda()
            .to(torch.bfloat16)
        )
        draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
        draft_model.freeze_embedding()
        print_with_rank(f"Loaded pre-trained base draft model.")

    # 根据策略冻结或添加LoRA
    if args.use_lora and args.lora_config:
        # for param in draft_model.parameters():
        #     param.requires_grad = False
        # print_with_rank(f"Frozen all base draft model parameters")
        
        # 为draft模型添加LoRA
        draft_lora_config = load_lora_config(args.lora_config, is_trainable=True)
        
        # PEFT compatibility fix: ensure prepare_inputs_for_generation is accessible
        if not hasattr(draft_model, 'prepare_inputs_for_generation'):
            print_with_rank("Warning: draft_model doesn't have prepare_inputs_for_generation, adding compatibility method")
            # Add a simple prepare_inputs_for_generation method from GenerationMixin
            from transformers.generation.utils import GenerationMixin
            if hasattr(GenerationMixin, 'prepare_inputs_for_generation'):
                draft_model.prepare_inputs_for_generation = GenerationMixin.prepare_inputs_for_generation.__get__(draft_model, draft_model.__class__)
        else:
            print_with_rank("Draft model has prepare_inputs_for_generation method")
        
        draft_model = get_peft_model(draft_model, draft_lora_config)
        draft_model = draft_model.to(torch.bfloat16)
        print_with_rank(f"Added new LoRA to draft model for training")
        
        # 详细记录所有模型参数的状态
        def log_model_parameters(model, model_name):
            """Log detailed parameter information for debugging"""
            print_with_rank(f"\n=== {model_name} Parameter Details ===")
            trainable_count = 0
            frozen_count = 0
            total_params = 0
            
            for name, param in model.named_parameters():
                trainable = param.requires_grad
                if trainable:
                    trainable_count += param.numel()
                else:
                    frozen_count += param.numel()
                total_params += param.numel()
                
                # 打印每个参数的详细信息
                print_with_rank(
                    f"  {name:60s} | "
                    f"Trainable: {str(trainable):5s} | "
                    f"Shape: {str(tuple(param.shape)):20s} | "
                    f"Dtype: {str(param.dtype):10s} | "
                    f"Device: {str(param.device):10s} | "
                    f"Params: {param.numel():,}"
                )
            
            print_with_rank(f"\n{model_name} Summary:")
            print_with_rank(f"  Total parameters: {total_params:,}")
            print_with_rank(f"  Trainable parameters: {trainable_count:,} ({100*trainable_count/total_params:.2f}%)")
            print_with_rank(f"  Frozen parameters: {frozen_count:,} ({100*frozen_count/total_params:.2f}%)")
            print_with_rank("=" * 80)
        
        # 记录target和draft模型的详细参数信息
        log_model_parameters(target_model, "Target Model")
        log_model_parameters(draft_model, "Draft Model")

        # # 从LoRA checkpoint恢复（只加载draft的LoRA）
        # if draft_model_last_checkpoint:
        #     draft_lora_path = os.path.join(draft_model_last_checkpoint, "draft_lora")
            
        #     if os.path.exists(draft_lora_path):
        #         draft_model.load_adapter(draft_lora_path, "default")
        #         print_with_rank(f"Loaded draft LoRA from checkpoint: {draft_lora_path}")


    
    print_with_rank(f"Initialized draft model")

    # build dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    # convert to dataloader
    cache_key = hashlib.md5(args.train_data_path.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    vocab_mapping_path = None
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        if not args.skip_vocab_mapping:
            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_eagle3_dataset,
                target_vocab_size=draft_model_config.vocab_size,
                draft_vocab_size=draft_model_config.draft_vocab_size,
                cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
                cache_key=cache_key,
            )
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized train dataloader")

    # we load the vocab mapping then
    if not args.skip_vocab_mapping:
        draft_model.load_vocab_mapping(vocab_mapping_path)
        print_with_rank(f"Loaded vocab mapping")


    if args.eval_data_path is not None:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            eval_dataset,
            tokenizer,
            args.chat_template,
            args.max_length,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
        )
        print_with_rank(f"Initialized eval dataloader")

    # build Eagle3 model
    # broadcast draft model
    eagle3_model = OnlineEagle3Model(
        target_model=target_model,
        draft_model=draft_model,
    )
    # eagle3_model = DDP(eagle3_model, find_unused_parameters=True)
    # target模型始终被忽略（无论是否使用LoRA，target都是冻结的）
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        ignored_modules=[target_model],
        process_group=get_dp_group(),
    )
    print_with_rank(f"Initialized Eagle3 FSDP model")
    
    # 打印参数统计信息
    if args.use_lora:
        print_trainable_parameters(target_model, "Target Model (Frozen)")
        print_trainable_parameters(draft_model, "Draft Model (LoRA Only)") 
        print_trainable_parameters(eagle3_model, "Eagle3 Model (Overall)")

    # build other components
    optimizer = torch.optim.AdamW(eagle3_model.parameters(), lr=args.learning_rate)
    
    if args.use_lora:
        # 统计LoRA参数数量用于日志记录
        lora_param_count = sum(1 for name, param in eagle3_model.named_parameters() 
                              if param.requires_grad and 'draft_model' in name and ('lora_' in name or 'adapter' in name))
        print_with_rank(f"Optimizer will train {lora_param_count} LoRA parameters out of total parameters")
    else:
        trainable_param_count = sum(1 for param in eagle3_model.parameters() if param.requires_grad)
        print_with_rank(f"Optimizer configured for {trainable_param_count} trainable parameters")
    
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )
    print_with_rank(f"Initialized optimizer and scheduler")

    # resume
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)

            optimizer.load_state_dict(state["optimizer_state_dict"])
            print_on_rank0("Successfully loaded optimizer state_dict.")

            scheduler.load_state_dict(state["scheduler_state_dict"])
            print_on_rank0("Successfully loaded scheduler state_dict.")

            start_epoch = state["epoch"] + 1
            print_on_rank0(f"Resuming from epoch {start_epoch}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )

    dist.barrier()

    # start running
    print_on_rank0(f"Starting training from epoch {start_epoch}")
    for epoch in range(start_epoch, args.num_epochs):
        # Run training
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        epoch_acces = [[] for _ in range(eagle3_model.module.length)]
        epoch_plosses = [[] for _ in range(eagle3_model.module.length)]

        for data in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
            optimizer.zero_grad()
            plosses, _, acces = eagle3_model(
                input_ids=data["input_ids"].cuda(),
                attention_mask=data["attention_mask"].cuda(),
                loss_mask=data["loss_mask"].cuda(),
            )

            # calculate weighted loss
            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            ploss.backward()
            if args.use_lora:
                grad_log_dict = {}
                lora_param_idx = 0
                for name, param in eagle3_model.named_parameters():
                    if param.requires_grad and 'draft_model' in name and ('lora_' in name or 'adapter' in name):
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            print_on_rank0(
                                f"LoRA param {lora_param_idx} ({name}) grad norm: {grad_norm}"
                            )
                            grad_log_dict[f"train/lora_grad_norm_{lora_param_idx}"] = grad_norm
                        lora_param_idx += 1
                if grad_log_dict:
                    wandb_log_if_initialized(grad_log_dict)
            optimizer.step()
            scheduler.step()

            logdict = {"train/lr": optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            wandb_log_if_initialized(logdict)

            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()
            wandb_log_if_initialized({f"train/epochacc_{i}": acc_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()
            wandb_log_if_initialized({f"train/epochploss_{i}": loss_i})
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )

        # run evaluation
        if args.eval_data_path is not None and epoch % args.eval_interval == 0:
            # Run evaluation
            draft_model.eval()
            eval_acces = [[] for _ in range(eagle3_model.length)]
            eval_plosses = [[] for _ in range(eagle3_model.length)]

            for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                plosses, _, acces = eagle3_model(
                    input_ids=data["input_ids"].cuda(),
                    attention_mask=data["attention_mask"].cuda(),
                    loss_mask=data["loss_mask"].cuda(),
                )
                eval_acces = [eval_acces[i] + [acces[i]] for i in range(len(acces))]
                eval_plosses = [
                    eval_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
                ]

            for i in range(len(eval_acces)):
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                dist.all_reduce(acc_i)
                acc_i = acc_i / dist.get_world_size()
                acc_i = acc_i.item()

                wandb_log_if_initialized({f"eval/epochacc_{i}": acc_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i},  Acc: {acc_i:.2f}"
                )

            for i in range(len(eval_plosses)):
                loss_i = torch.tensor(eval_plosses[i]).cuda().mean()
                dist.all_reduce(loss_i)
                loss_i = loss_i / dist.get_world_size()
                loss_i = loss_i.item()

                wandb_log_if_initialized({f"eval/epochploss_{i}": loss_i})
                print_on_rank0(
                    f"Eval Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.2f}"
                )

        if epoch % args.save_interval == 0:
            # Save the model
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()

            # Only gather a full state dict on rank 0 to reduce sync pressure
            with FSDP.state_dict_type(
                eagle3_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state_dict = eagle3_model.state_dict()
                state_to_save = {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }

                if dist.get_rank() == 0:
                    torch.save(
                        state_to_save,
                        os.path.join(epoch_output_dir, "training_state.pt"),
                    )
                    print_on_rank0(
                        f"Saved full training state to {epoch_output_dir}/training_state.pt"
                    )
                    
                    if args.use_lora:
                        # 手动提取并保存draft模型的LoRA权重（从FSDP state_dict中）
                        draft_lora_output_dir = os.path.join(epoch_output_dir, "draft_lora")
                        os.makedirs(draft_lora_output_dir, exist_ok=True)
                        
                        # 提取LoRA相关的权重
                        lora_state_dict = {}
                        for key, value in model_state_dict.items():
                            if "draft_model." in key and ("lora_" in key or "adapter" in key):
                                # 移除 "draft_model." 前缀，因为我们要保存的是draft模型内部的LoRA权重
                                lora_key = key.replace("draft_model.", "")
                                lora_state_dict[lora_key] = value
                        
                        if lora_state_dict:
                            # 保存LoRA权重
                            import safetensors.torch as st
                            st.save_file(lora_state_dict, os.path.join(draft_lora_output_dir, "adapter_model.safetensors"))
                            print_on_rank0(f"Saved {len(lora_state_dict)} LoRA weights to {draft_lora_output_dir}/adapter_model.safetensors")
                            
                            # 保存LoRA配置文件
                            import shutil
                            shutil.copy2(args.lora_config, os.path.join(draft_lora_output_dir, "adapter_config.json"))
                            print_on_rank0(f"Copied LoRA config to {draft_lora_output_dir}/adapter_config.json")
                        else:
                            print_on_rank0("Warning: No LoRA weights found in state_dict!")
                        
                        # 保存tokenizer到draft_lora目录
                        tokenizer.save_pretrained(draft_lora_output_dir)
                        print_on_rank0(f"Saved tokenizer to {draft_lora_output_dir}")
                    else:
                        # 原始逻辑：保存draft模型状态
                        draft_model_state_dict = {
                            k.replace("draft_model.", ""): v
                            for k, v in model_state_dict.items()
                            if "draft_model." in k
                        }
                        draft_model.save_pretrained(
                            epoch_output_dir,
                            state_dict=draft_model_state_dict,
                        )
                        print_on_rank0(f"Saved model configuration to {epoch_output_dir}")
                        
                        # 保存tokenizer
                        tokenizer.save_pretrained(epoch_output_dir)
                        print_on_rank0(f"Saved tokenizer to {epoch_output_dir}")
                    # Avoid a trailing barrier here to reduce chances of hanging on sync

    destroy_distributed()


if __name__ == "__main__":
    main()