from sgl_spec.data.dataloader import prepare_dataloaders
from sgl_spec.data.config import DataConfig
from typing import Optional, Tuple, List
from datasets import load_dataset
from collections import Counter
import torch
import os


def process_token_dict_to_mappings(
    token_dict: Counter, 
    draft_vocab_size: int, 
    vocab_size: int, 
    load_from_cache_file: str = ''
) -> Tuple[List[int], List[bool]]:
    """
    Process token_dict to create d2t and t2d mappings, with optional caching.
    """
   
    if load_from_cache_file and os.path.exists(load_from_cache_file):
        print(f"Loading d2t/t2d from cache: {load_from_cache_file}")
        data = torch.load(load_from_cache_file)
        return data['d2t'], data['t2d']

    total_frequency = sum(token_dict.values())
    top_N = token_dict.most_common(draft_vocab_size)
    top_N_frequency_sum = sum(freq for key, freq in top_N)
    top_N_ratio = top_N_frequency_sum / total_frequency
    print(f"top {draft_vocab_size} token frequency ratio: {top_N_ratio:.2%}")
    
    used_tokens = [key for key, freq in top_N]
    used_tokens.sort()
    used_tokens_set = set(used_tokens)
    
    # Create d2t mapping: draft token index -> target token index
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    
    # Create t2d mapping: target token index -> boolean (whether in draft vocab)
    t2d = [i in used_tokens_set for i in range(vocab_size)]

    if load_from_cache_file:
        # Create the directory for the cache file if it doesn't exist
        cache_dir = os.path.dirname(load_from_cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save({'d2t': d2t, 't2d': t2d}, load_from_cache_file)
        print(f"Saved d2t/t2d to cache: {load_from_cache_file}")
    
    return load_from_cache_file


def prepare_full_dataloaders(
    tokenizer, 
    train_data_path: str, 
    test_data_path: str, 
    draft_model=None,
    config: Optional[DataConfig] = None,
):
    if config is None:
        config = DataConfig()
    
    train_data = load_dataset("json", data_files=train_data_path)['train']
    test_data = load_dataset("json", data_files=test_data_path)['train']
    
    train_loader, test_loader, train_sampler, test_sampler, token_dict = prepare_dataloaders(
        train_data, test_data, tokenizer, config
    )
    
    # Process token_dict if draft_model is provided
    d2t = None
    t2d = None
    if draft_model is not None and token_dict is not None:
        draft_vocab_size = draft_model.draft_vocab_size
        vocab_size = draft_model.vocab_size
        _load_from_cache_file = getattr(config, 'load_from_cache_file', False)
        d2t_path = process_token_dict_to_mappings(
            token_dict, draft_vocab_size, vocab_size,load_from_cache_file=_load_from_cache_file
        )
        # Update draft model's vocab buffers
        draft_model.d2t.copy_(torch.tensor(d2t, dtype=torch.int64))
        draft_model.t2d.copy_(torch.tensor(t2d, dtype=torch.bool))
    
    return train_loader, test_loader, train_sampler, test_sampler, d2t_path