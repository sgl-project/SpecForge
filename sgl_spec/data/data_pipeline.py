from sgl_spec.data.dataloader import prepare_dataloaders
from sgl_spec.data.config import DataConfig
from typing import Optional
from datasets import load_dataset

def prepare_full_dataloaders(tokenizer, train_data_path: str, test_data_path: str, config: Optional[DataConfig] = None):
    if config is None:
        config = DataConfig()
    
    train_data = load_dataset("json", data_files=train_data_path)['train']
    test_data = load_dataset("json", data_files=test_data_path)['train']
    return prepare_dataloaders(train_data, test_data, tokenizer, config)