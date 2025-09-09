import argparse
import hashlib
import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from accelerate.utils import set_seed
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
from specforge.data import build_eagle3_dataset, generate_vocab_mapping_file
from specforge.data.utils import DataCollatorWithPadding
from specforge.utils import get_last_checkpoint, print_with_rank, rank_0_priority

import deepspeed


def save_draft_model_safetensors(
    model_engine,
    output_dir: str,
    exclude_substr=("embed",),
    shard_size="2GB",
):
    if dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    dist.barrier()

    draft = model_engine.module.draft_model
    params_to_gather = [
        p for n, p in draft.named_parameters()
        if not any(s in n.lower() for s in exclude_substr)
    ]

    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0), torch.no_grad():
        if dist.get_rank() == 0:
            full_sd = {
                k: v.detach().cpu()
                for k, v in draft.state_dict().items()
                if not any(s in k.lower() for s in exclude_substr)
            }
            draft.save_pretrained(
                output_dir,
                state_dict=full_sd,
                safe_serialization=True,   # -> safetensors
                max_shard_size=shard_size
            )
            print(f"[rank0] Draft model (filtered) saved to: {output_dir}")

    dist.barrier()


def get_zero_config(args,total_steps,warmup_steps):
    zero_stages = {
        "0":{
            "zero_optimization": {
                "stage": 0,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "overlap_comm": False,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True,
                "round_robin_gradients": True
            }
        },
        "1":{
            "zero_optimization": {
                "stage": 1,
                "overlap_comm": True,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "contiguous_gradients": True
            }
        },
        "2":{
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "round_robin_gradients":True,
                "allgather_partitions": True,
            }
        },
        "3":{
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "contiguous_gradients": True,
                "stage3_gather_16bit_weights_on_model_save":True,
                "stage3_max_live_parameters" : 1e9,
                "stage3_param_persistence_threshold": 1e6,
                # "zero_hpz_partition_size": 8,
                "stage3_max_reuse_distance" : 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size" : 5e8,
            }
        }
    }
    ds_config = { 
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": {
            "enabled": True,
            "immediate_grad_update": True,
            "check_grad_overflow": True
        },
        "fp16": {
            "enabled": False
        },
        "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": args.learning_rate,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95]
        }
        },
        "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": total_steps,
            "warmup_num_steps": warmup_steps,
            "warmup_min_lr":0,
            "warmup_max_lr": args.learning_rate
            }
        },
        **zero_stages[str(args.zero_stage)],
        "communication_data_type": "bf16",
    }

    return ds_config
  
def parse_args():
    parser = argparse.ArgumentParser(description="Train Eagle3 with online data")
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--draft-model-config", type=str, required=True)
    parser.add_argument(
        "--embedding-key",
        type=str,
        default="model.embed_tokens.weight",
        help="The key of the embedding weight to load from the target model",
    )
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ttt-length",
        type=int,
        default=7,
        help="The length for Test-Time Training (TTT).",
    )
    parser.add_argument("--chat-template", type=str, default="llama3")
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
    parser.add_argument("--attention-backend", type=str, default="flex_attention")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0) 
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1) 
    parser.add_argument("--zero-stage", type=int, default=3) 
    args = parser.parse_args()
    return args


def print_on_rank0(message):
    if dist.get_rank() == 0:
        print(message)


def main():
    args = parse_args()
    set_seed(args.seed)

    deepspeed.init_distributed()

    local_rank = int(os.getenv("LOCAL_RANK",0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 8))
    torch.cuda.set_device(local_rank)
    print_with_rank(f"WORLD SIZE={world_size}-RANK={rank}-LOCAL_WORLD_SIZE={local_world_size}-LOCAL RANK={local_rank}")
    print("Initialized distributed environment")

    draft_model_last_checkpoint = None
    if args.resume and os.path.isdir(args.output_dir):
        print_on_rank0(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    if draft_model_last_checkpoint:
        draft_model = (
            AutoEagle3DraftModel.from_pretrained(
                draft_model_last_checkpoint, attention_backend=args.attention_backend
            )
            .cuda()
            .to(torch.bfloat16)
        )
    else:
        draft_model = (
            AutoEagle3DraftModel.from_config(
                draft_model_config, attention_backend=args.attention_backend
            )
            .cuda()
            .to(torch.bfloat16)
        )
    
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    draft_model.freeze_embedding()
    print_with_rank("Initialized draft model")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    with rank_0_priority():
        train_eagle3_dataset = build_eagle3_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
        )
        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    train_sampler = DistributedSampler(
        train_eagle3_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )

    train_dataloader = DataLoader(
        train_eagle3_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=DataCollatorWithPadding(),
        persistent_workers=True,
    )

    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print_with_rank("Initialized train dataloader")

    ds_config = get_zero_config(args,total_steps,warmup_steps)
    dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_path, torch_dtype=torch.bfloat16)

    # https://github.com/deepspeedai/DeepSpeed/issues/7461
    for m in target_model.modules():
        if "SparseMoeBlock" in m.__class__.__name__:
            deepspeed.utils.set_z3_leaf_modules(target_model, [m.__class__])
            print(f"--------Setting zero3 leaf for model on class with name: {m.__class__.__name__}---------")
            break
    
    for _, param in target_model.named_parameters():
        param.requires_grad = False
    print_with_rank("Initialized target model")

    eagle3_model = OnlineEagle3Model(
        target_model= target_model,
        draft_model = draft_model,
        length = args.ttt_length,
        attention_backend = args.attention_backend,
    )
    print_with_rank("Initialized Eagle3 model")
    
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=eagle3_model,
        config=ds_config,
        model_parameters=eagle3_model.draft_model.parameters()
    )
    print_with_rank("Initialized Eagle3 DeepSpeed model")

    print_with_rank("Start Training!")
    start_epoch = 0
    if draft_model_last_checkpoint is not None:
        print_on_rank0(
            f"Resuming draft model training from checkpoint: {draft_model_last_checkpoint}"
        )
        state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)

            try:
                model_engine.optimizer.load_state_dict(state["optimizer_state_dict"])
                print_on_rank0("Successfully loaded optimizer state_dict.")
            except:
                print_on_rank0("Warning: Failed to load optimizer state_dict.")

            try:
                scheduler.load_state_dict(state["scheduler_state_dict"])
                print_on_rank0("Successfully loaded scheduler state_dict.")
            except:
                print_on_rank0("Warning: Failed to load scheduler state_dict.")

            start_epoch = state["epoch"] + 1
            print_on_rank0(f"Resuming from epoch {start_epoch}")
        else:
            print_on_rank0(
                f"Warning: Checkpoint directory {draft_model_last_checkpoint} found, but training_state.pt is missing. Starting from scratch."
            )

    dist.barrier()
    print_on_rank0(f"Starting training from epoch {start_epoch}")

    global_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model_engine.train()
        draft_model.train()  # for consistency

        epoch_acces = [[] for _ in range(model_engine.module.length)]
        epoch_plosses = [[] for _ in range(model_engine.module.length)]

        # Training loop
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        for data in pbar:
            input_ids = data["input_ids"].to(model_engine.device)
            attention_mask = data["attention_mask"].to(model_engine.device)
            loss_mask = data["loss_mask"].to(model_engine.device)

            # Forward
            plosses, _, acces = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
            )
            # Weighted loss
            ploss_weight = [0.8 ** i for i in range(len(plosses))]
            ploss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))

            # Backward
            model_engine.backward(ploss)

            model_engine.step()  # zero_grad, optimizer.step, scheduler.step, grad clip

            global_step += 1

            # Accumulate for epoch logging
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

            # Update progress bar
            pbar.set_postfix({"ploss": ploss.item(), "lr": model_engine.get_lr()[0]})

        # Epoch-level logging
        for i in range(len(epoch_acces)):
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            dist.all_reduce(acc_i)
            acc_i = acc_i / dist.get_world_size()
            acc_i = acc_i.item()
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, Acc: {acc_i:.4f}"
            )

        for i in range(len(epoch_plosses)):
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            dist.all_reduce(loss_i)
            loss_i = loss_i / dist.get_world_size()
            loss_i = loss_i.item()
            print_on_rank0(
                f"Train Epoch [{epoch + 1}/{args.num_epochs}], position {i}, pLoss: {loss_i:.4f}"
            )

        # Save checkpoint
        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            if dist.get_rank() == 0:
                os.makedirs(epoch_output_dir, exist_ok=True)
            dist.barrier()
            # Only save full state on rank 0
            if dist.get_rank() == 0:
                # Save training state
                state_to_save = {
                    "optimizer_state_dict": model_engine.optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                torch.save(
                    state_to_save,
                    os.path.join(epoch_output_dir, "training_state.pt"),
                )
                print_on_rank0(f"Saved training state to {epoch_output_dir}/training_state.pt")

            save_draft_model_safetensors(model_engine, epoch_output_dir)

        dist.barrier()

    print_on_rank0("Training completed.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()