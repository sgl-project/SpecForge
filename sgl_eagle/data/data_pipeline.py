import os
import subprocess
import importlib
import inspect
import random
from sgl_eagle.data.data_convert import ShareGPTProcessor, Ultrachat200KProcessor
from sgl_eagle.data.dataloader import prepare_dataloaders
from sgl_eagle.data.config import DataConfig
from datasets import concatenate_datasets
from typing import Optional

#TODO opt:All memory processing, possible oom.
def prepare_full_dataloaders(tokenizer, temp_dir="/data/eagle_data", config: Optional[DataConfig] = None):
    if config is None:
        config = DataConfig()
    
    all_data = []
    #TODO Support configuring different datasets. 
    for processor_cls in [ShareGPTProcessor, Ultrachat200KProcessor]:
        processor = processor_cls()
        data = processor.process()
        all_data.append(data)
    all_data = concatenate_datasets(all_data)
    split = all_data.train_test_split(test_size=config.test_size)
    train_data = split['train']
    test_data = split['test']
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    return prepare_dataloaders(train_data, test_data, tokenizer, config)