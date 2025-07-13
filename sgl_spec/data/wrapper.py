"""

import torch
from sgl_spec.data.dataset_build import scandata, do_word_count

class EagleDatasetWrapper:
    def __init__(self, ds_path, max_len=2048):
        self.ds = scandata(ds_path, max_len)
        N = self.draft_vocab_size
        token_dict = do_word_count(self.ds)
        print("finished do_word_count")
        total_frequency = sum(token_dict.values())
        top_N = token_dict.most_common(N)
        top_N_frequency_sum = sum(freq for key, freq in top_N)
        top_N_ratio = top_N_frequency_sum / total_frequency
        print(f"top {N} token frequency ratio: {top_N_ratio:.2%}")
        used_tokens = [key for key, freq in top_N]
        used_tokens.sort()
        used_tokens_set = set(used_tokens)
        self.d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
        self.t2d = [i in used_tokens_set for i in range(self.vocab_size)]
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        item["input_ids"] = torch.tensor(item["input_ids"])
        item["loss_mask"] = torch.tensor(item["loss_mask"])
        item["hidden_state"] = torch.tensor(item["hidden_state"])
        item["target_hidden_states"] = torch.tensor(item["target_hidden_states"])
        return item

"""
