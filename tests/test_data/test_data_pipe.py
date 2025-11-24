import hashlib
import json
import os
import pickle
import time

import psutil
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from specforge.data.config import DataConfig, ModelType
from specforge.data.data_pipeline import prepare_full_dataloaders


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_cache_key(config, model_path):
    """Generate cache key based on configuration"""
    cache_data = {
        "model_path": model_path,
        "model_type": config.model_type.value,
        "batch_size": config.batch_size,
        "num_processes": config.num_processes,
        "max_length": config.max_length,
        "test_size": config.test_size,
        "shuffle_seed": config.shuffle_seed,
        "load_from_cache_file": config.load_from_cache_file,
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


def save_dataloader_cache(
    cache_key,
    train_loader,
    test_loader,
    train_sampler,
    test_sampler,
    token_dict,
    cache_dir,
):
    """Save dataloader cache"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"dataloader_cache_{cache_key}.pkl")

    # Only save key information, not the entire loader objects
    cache_data = {
        "train_dataset": train_loader.dataset,
        "test_dataset": test_loader.dataset,
        "token_dict": token_dict,
        "train_sampler_state": {
            "num_replicas": train_sampler.num_replicas,
            "rank": train_sampler.rank,
            "shuffle": train_sampler.shuffle,
        },
        "test_sampler_state": {
            "num_replicas": test_sampler.num_replicas,
            "rank": test_sampler.rank,
            "shuffle": test_sampler.shuffle,
        },
        "config_batch_size": train_loader.batch_size,
        "config_num_workers": train_loader.num_workers,
        "config_pin_memory": train_loader.pin_memory,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f)

    print(f"Cache saved to: {cache_file}")


def load_dataloader_cache(cache_key, config, cache_dir):
    """Load dataloader cache"""
    cache_file = os.path.join(cache_dir, f"dataloader_cache_{cache_key}.pkl")

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        # Recreate DataLoader objects
        from torch.utils.data import DataLoader, DistributedSampler

        from specforge.data.dataloader import DataCollatorWithPadding

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        train_sampler = DistributedSampler(
            cache_data["train_dataset"],
            num_replicas=world_size,
            rank=rank,
            shuffle=cache_data["train_sampler_state"]["shuffle"],
        )

        test_sampler = DistributedSampler(
            cache_data["test_dataset"],
            num_replicas=world_size,
            rank=rank,
            shuffle=cache_data["test_sampler_state"]["shuffle"],
        )

        train_loader = DataLoader(
            cache_data["train_dataset"],
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=DataCollatorWithPadding(),
        )

        test_loader = DataLoader(
            cache_data["test_dataset"],
            batch_size=config.batch_size,
            sampler=test_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=DataCollatorWithPadding(),
        )

        print(f"Successfully loaded from cache: {cache_file}")
        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            cache_data["token_dict"],
        )

    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


def main():
    print("Starting data pipeline test...")

    # Initialize distributed environment (single machine mode)
    if not dist.is_initialized():
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        # Initialize process group
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )
        print("Distributed environment initialized (single machine mode)")

    # Configuration
    model_path = "/data/eagle_data/shenggui/models/Llama-4-Scout-17B-16E-Instruct"
    temp_dir = "/data/eagle_data/chao"
    cache_dir = os.path.join(temp_dir, "dataloader_cache")

    # Create configuration object
    config = DataConfig(
        model_type=ModelType.LLAMA4,
        max_length=2048,
        load_from_cache_file=True,  # Enable cache
    )

    print(f"Configuration:")
    print(f"  - Model type: {config.model_type.value}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Number of processes: {config.num_processes}")
    print(f"  - Max length: {config.max_length}")
    print(f"  - Cache enabled: {config.load_from_cache_file}")

    # Get chat template information
    template = config.get_chat_template()
    print(f"  - Assistant Header: {repr(template['assistant_header'])}")
    print(f"  - User Header: {repr(template['user_header'])}")

    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    print(f"\nStart time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Start memory: {start_memory:.2f} MB")

    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer_time = time.time() - tokenizer_start
        print(f"Tokenizer loaded, time taken: {tokenizer_time:.2f}s")

        # Generate cache key
        cache_key = get_cache_key(config, model_path)
        print(f"Cache key: {cache_key}")

        # Try to load from cache
        print("\nChecking cache...")
        cached_result = load_dataloader_cache(cache_key, config, cache_dir)

        if cached_result is not None:
            train_loader, test_loader, train_sampler, test_sampler, token_dict = (
                cached_result
            )
            print("Data pipeline loaded from cache successfully!")
            dataloader_time = 0  # Loading from cache is very fast
        else:
            # Prepare data loaders
            print("\nPreparing data loaders...")
            dataloader_start = time.time()
            train_loader, test_loader, train_sampler, test_sampler, token_dict = (
                prepare_full_dataloaders(tokenizer, temp_dir=temp_dir, config=config)
            )
            dataloader_time = time.time() - dataloader_start

            # Save to cache
            print("\nSaving cache...")
            save_dataloader_cache(
                cache_key,
                train_loader,
                test_loader,
                train_sampler,
                test_sampler,
                token_dict,
                cache_dir,
            )

        # Record end time and memory
        end_time = time.time()
        end_memory = get_memory_usage()
        total_time = end_time - start_time

        print(f"\nData pipeline preparation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Data loader preparation time: {dataloader_time:.2f}s")
        print(f"End memory: {end_memory:.2f} MB")
        print(f"Memory increase: {end_memory - start_memory:.2f} MB")

        # Dataset information
        print(f"\nDataset information:")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Token dictionary size: {len(token_dict) if token_dict else 0}")

        # Show top 10 most common tokens
        if token_dict:
            print(f"\nTop 10 most common tokens:")
            for i, (token_id, count) in enumerate(token_dict.most_common(10)):
                try:
                    token_text = (
                        tokenizer.decode([token_id])
                        if hasattr(tokenizer, "decode")
                        else str(token_id)
                    )
                    print(f"  {i+1}. Token {token_id} ('{token_text}'): {count}")
                except:
                    print(f"  {i+1}. Token {token_id}: {count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nTest completed!")

    # Clean up distributed environment
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed environment cleaned up")


if __name__ == "__main__":
    main()
