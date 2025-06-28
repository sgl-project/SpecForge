import sys
sys.path.append("/workspace/model-performance/yikaizhu/sgl-spec")
import json
import os
import torch
import torch.distributed as dist

import wandb
import random
import argparse

from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer, AutoConfig
from transformers.training_args import TrainingArguments

from sgl_eagle.modeling.draft.llama_eagle import LlamaForCausalLMEagle
from sgl_eagle.data.offline_eagle_data_wrapper import (
    EagleLocalDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    list_local_files,
)
from sgl_eagle.core.offline_trainer_eagle_ttt import OfflineEagleTrainTimeTestTrainer

# Initialize wandb only on rank 0 to avoid conflicts
wandb_run_name = "06-27-2025-Qwen2.5-7B-Instruct-EAGLE-TTT"
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank <= 0:
    wandb.init(project="BaldEagle", mode="offline", name=wandb_run_name)
else:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=os.environ["MODEL_PATH"])
    parser.add_argument("--sharegpt-datapaths", type=str, default=os.environ["SHAREGPT_DATAPATHS"])
    parser.add_argument("--ultra-chat-datapaths", type=str, default=os.environ["ULTRACHAT_DATAPATHS"])
    parser.add_argument("--output-dir", type=str, default=f"./hf_trainer_output_dir/{wandb_run_name}")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    model_path = args.model_path
    sharegpt_datapaths = args.sharegpt_datapaths
    ultra_chat_datapaths = args.ultra_chat_datapaths

    # -------------------------------- Load original Llama weights --------------------------------

    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
        lm_head_path = index_json["weight_map"]["lm_head.weight"]

    with safe_open(os.path.join(model_path, emb_path), framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice("model.embed_tokens.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].to(torch.bfloat16)

    with safe_open(os.path.join(model_path, lm_head_path), framework="pt", device="cpu") as f:
        lm_head_weights = f.get_slice("lm_head.weight")[:, :].to(torch.bfloat16)


    # -------------------------------- Create draft model + tokenizer + head --------------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Get the device for DDP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_args = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=1,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        num_key_value_heads=config.num_key_value_heads,
        num_attention_heads=config.num_attention_heads,
        tie_word_embeddings=config.tie_word_embeddings,
        torch_dtype=config.torch_dtype,
        draft_vocab_size=config.vocab_size,
    )

    draft_model = LlamaForCausalLMEagle(model_args)
    draft_model.load_embedding_weights(tensor)
    # Ensure all parameters are bfloat16
    for param in draft_model.parameters():
        param.data = param.data.to(torch.bfloat16)
    draft_model.embed_tokens.requires_grad_(False)  # Use in-place operation
    draft_model = draft_model.to(device)

    # Load head
    head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False, dtype=torch.bfloat16)
    head.weight.data = lm_head_weights
    head = head.to(device)  # Move head to device for DDP
    head.eval()
    for param in head.parameters():
        param.requires_grad = False

    # -------------------------------- Load data --------------------------------

    sharegpt_datapaths = list_local_files(sharegpt_datapaths)
    ultra_chat_datapaths = list_local_files(ultra_chat_datapaths)

    combined_data_paths = (
        sharegpt_datapaths[: int(len(sharegpt_datapaths) * 0.95)] + ultra_chat_datapaths
    )
    random.Random(42).shuffle(combined_data_paths)
    eval_data_paths = sharegpt_datapaths[int(len(sharegpt_datapaths) * 0.95) :][:100]

    eagle_train_dataset = EagleLocalDataset(
        combined_data_paths, transform=AddUniformNoise(std=0.5)
    )
    eagle_test_dataset = EagleLocalDataset(eval_data_paths)

    eagle_collator = DataCollatorWithPadding()

    # -------------------------------- Train --------------------------------

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        ddp_find_unused_parameters=False,  # Set to True if your model has unused parameters
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        warmup_ratio=0.01,
        learning_rate=1e-4,  # 1e-3
        lr_scheduler_type="constant",  # Placeholder, we override it in the trainer
        max_grad_norm=0.5,  # 1
        adam_beta1=0.9,  # 0.9
        adam_beta2=0.95,  # 0.999
        weight_decay=1e-2,
        eval_strategy="steps",
        eval_steps=256,
        save_strategy="steps",
        save_steps=0.1,  # saves every 10% of training
        save_total_limit=3,
        report_to=["wandb"] if local_rank <= 0 else [],  # Only log to wandb on rank 0
        log_on_each_node=False,  # Only log on main process
        logging_dir=f'{args.output_dir}/logs',  # TensorBoard log dir
    )

    trainer = OfflineEagleTrainTimeTestTrainer(
        model=draft_model,
        head=head,
        args=training_args,
        train_dataset=eagle_train_dataset,
        eval_dataset=eagle_test_dataset,
        data_collator=eagle_collator,
        min_lr_ratio=0.5,  # Custom lr scheduler param
    )

    trainer.train()

    print(f"Rank {local_rank}: Saving model...")
    trainer.save_model()
    print(f"Rank {local_rank}: Model saved")

    # Synchronize after saving but before hub operations
    if dist.is_initialized():
        print(f"Rank {local_rank}: Synchronizing after save...")
        dist.barrier()

    # Only push to hub and finish wandb on rank 0
    if local_rank <= 0:
        wandb.finish()
        print("Rank 0: Wandb finished")
    else:
        print(f"Rank {local_rank}: Waiting for rank 0 to complete hub operations...")
