from typing import Any, Optional, Union


import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache



def _static_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_position is None:
        # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
        k_cache.copy_(key_states)
        v_cache.copy_(value_states)
    else:
        # Generation phase. Update specific positions.
        # Use index_copy_ for in-place update (compile-friendly).
        try:
            k_cache.index_copy_(2, cache_position, key_states)
            v_cache.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            k_cache[:, :, cache_position] = key_states
            v_cache[:, :, cache_position] = value_states
    return k_cache, v_cache




class StaticCache(Cache):
    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len =  max_cache_len

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.config = config
        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        device = torch.device(device) if device is not None else None
        self.device = device
        self.current_length_data = None
        self.past_key_values_data_list = None

        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self._dtype, device=layer_device)
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

        self.init_cache()

    def init_cache(self):
        config = self.config
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.current_length_data = torch.zeros(config.num_hidden_layers * 2, dtype=torch.long)

        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )

        # Allocate a unified tensor for all key/value pairs: [2 * num_layers, B, H, T, D]
        self.past_key_values_data = torch.zeros(
            config.num_hidden_layers * 2,
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
            device=self.device,
            dtype=self._dtype,
        )
        self.past_key_values_data_list = [self.past_key_values_data]  # still a list for compatibility

        for idx in range(config.num_hidden_layers):
            k = self.past_key_values_data[2 * idx]
            v = self.past_key_values_data[2 * idx + 1]

            torch._dynamo.mark_static_address(k)
            torch._dynamo.mark_static_address(v)

            self.key_cache.append(k)
            self.value_cache.append(v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if cache_kwargs is None:
            cache_kwargs = {}

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)

        cache_position = cache_kwargs.get("cache_position")

        k_updated, v_updated = _static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_position,
        )

        kv_tensor = self.past_key_values_data_list[0]
        offset = layer_idx * 2
        if cache_position is None:
            kv_tensor[offset].copy_(key_states)
            kv_tensor[offset + 1].copy_(value_states)
        else:
            kv_tensor[offset].index_copy_(2, cache_position, key_states)
            kv_tensor[offset + 1].index_copy_(2, cache_position, value_states)

            # Update current length data
            max_pos = cache_position.max().item() + 1
            self.current_length_data[offset] = max_pos
            self.current_length_data[offset + 1] = max_pos
        return k_updated, v_updated

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_mask_sizes(self, cache_position, layer_idx) -> tuple[int, int]:
        kv_length = self.get_max_cache_shape()
        return kv_length, 0

    def get_meta(self):
        return self.past_key_values_data_list, self.current_length_data

    def sync_from_past_data(self):
        """Ensure key_cache and value_cache are synced from past_key_values_data."""
        for i in range(self.config.num_hidden_layers):
            self.key_cache[i] = self.past_key_values_data[2 * i]
            self.value_cache[i] = self.past_key_values_data[2 * i + 1]



def initialize_past_key_values(model, max_length=2048):
    past_key_values = StaticCache(
        config=model.target_model.config, 
        max_batch_size=1, 
        max_cache_len=max_length,
        device="cuda",
        dtype=torch.float16
        )
    return past_key_values


if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("cache/model/llava-1.5-7b-hf")

    cache = StaticCache(config=config, max_cache_len=2048, device="cuda",max_batch_size=1)