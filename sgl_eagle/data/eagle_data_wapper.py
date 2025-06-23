import torch
from datasets import concatenate_datasets, load_from_disk
from torch.nn.utils.rnn import pad_sequence



def custom_eagle_collate_fn(batch):

    input_ids = [item["input_ids"] for item in batch]
    loss_mask = [item["loss_mask"] for item in batch]
    hidden_state = [item["hidden_state"] for item in batch]
    target_hidden_states = [item["target_hidden_states"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    loss_mask = pad_sequence(loss_mask, batch_first=True, padding_value=0)
    hidden_state = pad_sequence(hidden_state, batch_first=True, padding_value=0)
    target_hidden_states = pad_sequence(target_hidden_states, batch_first=True, padding_value=0)


    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_state": hidden_state,
        "target_hidden_states": target_hidden_states
    }

class EagleDatasetWrapper:
    def __init__(self, ds_path,max_len=2048):
        ds_list=[]
        chunk_dirs = [os.path.join(ds_path, d) for d in os.listdir(ds_path) if d.startswith("chunk_")]
        ds_list = [load_from_disk(chunk_dir) for chunk_dir in sorted(chunk_dirs)]
        self.ds=concatenate_datasets(ds_list)
        self.max_len=max_len

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        # 这里可以加上 tensor 转换
        item["input_ids"] = torch.tensor(item["input_ids"])
        item["loss_mask"] = torch.tensor(item["loss_mask"])
        item["hidden_state"] = torch.tensor(item["hidden_state"])
        item["target_hidden_states"] = torch.tensor(item["target_hidden_states"])
        return item