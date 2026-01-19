import torch
from contextlib import nullcontext

class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
    ):
        device = "cpu"
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device=device)
            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_token_logprobs_idx = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_top_logprobs_val = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.float32, device=device
            )
            self.output_top_logprobs_idx = torch.zeros(
                (size, max_top_logprobs_num), dtype=torch.int32, device=device
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = torch.zeros(
                (size, hidden_size), dtype=hidden_states_dtype, device=device
            )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )