import os
import pickle
import tempfile
from collections import Counter
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from sgl_spec.data.config import DataConfig
from sgl_spec.data.data_utils import preprocess_conversations


def do_word_count(dataset, num_processes: int = 8, config: Optional[DataConfig] = None):
    if config is None:
        config = DataConfig()

    def _process_data(rows):
        # Calculate total length to avoid repeated concatenation
        total_length = sum(len(ids) for ids in rows["input_ids"])
        if total_length == 0:
            return {"unique_ids": [[]], "counts": [[]]}

        # Pre-allocate tensors for better performance
        all_input_ids = torch.zeros(total_length, dtype=torch.long)
        all_loss_mask = torch.zeros(total_length, dtype=torch.long)

        # Fill tensors
        offset = 0
        for ids, mask in zip(rows["input_ids"], rows["loss_mask"]):
            length = len(ids)
            if length > 0:
                all_input_ids[offset : offset + length] = torch.LongTensor(ids)
                all_loss_mask[offset : offset + length] = torch.LongTensor(mask)
                offset += length

        masked_ids = all_input_ids[all_loss_mask == 1]
        if len(masked_ids) == 0:
            return {"unique_ids": [[]], "counts": [[]]}

        unique_ids, counts = masked_ids.unique(return_counts=True)
        return {"unique_ids": [unique_ids.tolist()], "counts": [counts.tolist()]}

    new_ds = dataset.map(
        _process_data,
        batched=True,
        batch_size=config.preprocess_batch_size,
        num_proc=num_processes,
        remove_columns=dataset.column_names,
    )

    token_dict = Counter()
    for row in new_ds:
        if (
            row["unique_ids"]
            and len(row["unique_ids"]) > 0
            and len(row["unique_ids"][0]) > 0
        ):
            token_dict.update(dict(zip(row["unique_ids"][0], row["counts"][0])))
    return token_dict


def build_dataset_rank(
    tokenizer=None,
    ds=None,
    assistant_header: Optional[str] = None,
    user_header: Optional[str] = None,
    max_length: Optional[int] = None,
    compute_token_dict: bool = True,
    config: Optional[DataConfig] = None,
):

    if ds is None:
        raise ValueError("ds parameter cannot be None")

    if config is None:
        config = DataConfig()

    if max_length is None:
        max_length = config.max_length

    # Get chat template from config
    template = config.get_chat_template()
    if assistant_header is None:
        assistant_header = template["assistant_header"]
    if user_header is None:
        user_header = template["user_header"]

    ds = ds.shuffle(seed=config.shuffle_seed)
    original_cols = ds.column_names
    num_proc = config.num_processes

    # Create a temporary file to collect token statistics during processing
    if compute_token_dict:
        token_stats_filename = tempfile.mktemp()
        token_stats_file_prefix = token_stats_filename
    else:
        token_stats_filename = None
        token_stats_file_prefix = None

    def preprocess_function(examples):
        # Always do preprocessing
        processed = preprocess_conversations(
            tokenizer,
            examples["conversations"],
            return_attention_mask=True,
            assistant_header=assistant_header,
            user_header=user_header,
            max_length=max_length,
            config=config,
        )

        if compute_token_dict and token_stats_file_prefix:
            # Vectorized tensor processing
            all_input_ids = torch.cat(
                [torch.LongTensor(item[0]) for item in processed["input_ids"]], dim=0
            )
            all_loss_masks = torch.cat(
                [torch.LongTensor(item[0]) for item in processed["loss_mask"]], dim=0
            )

            masked_ids = all_input_ids[all_loss_masks == 1]

            if len(masked_ids) > 0:
                unique_ids, counts = masked_ids.unique(return_counts=True)
                batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))

                pid = os.getpid()
                with open(f"{token_stats_file_prefix}.{pid}", "ab") as f:
                    pickle.dump(batch_token_dict, f)

        return processed

    # Process dataset only once
    dataset = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_cols,
        load_from_cache_file=config.load_from_cache_file,
    )

    # Collect token dictionary from file if needed
    token_dict = None
    if compute_token_dict and token_stats_file_prefix:
        import glob

        token_dict = Counter()
        for fname in glob.glob(f"{token_stats_file_prefix}.*"):
            try:
                with open(fname, "rb") as f:
                    while True:
                        try:
                            batch_dict = pickle.load(f)
                            token_dict.update(batch_dict)
                        except EOFError:
                            break
            except FileNotFoundError:
                pass
            finally:
                if os.path.exists(fname):
                    os.unlink(fname)

    dataset.set_format(type="torch")

    return dataset, token_dict


def build_test_dataset(
    tokenizer=None,
    ds=None,
    assistant_header: Optional[str] = None,
    user_header: Optional[str] = None,
    max_length: Optional[int] = None,
    config: Optional[DataConfig] = None,
):
    """
    Simplified version for test data that doesn't need token counting
    """
    if config is None:
        config = DataConfig()

    if max_length is None:
        max_length = config.max_length

    return build_dataset_rank(
        tokenizer,
        ds,
        assistant_header,
        user_header,
        max_length,
        compute_token_dict=False,
        config=config,
    )
