"""Qwen3 full-prefill target K/V capture for SpecForge's Mooncake sink."""

import torch

_geometry = None


def capture_metadata(spec):
    requested = spec.get("target_kv_geometry")
    if requested is None and _geometry is None:
        return {}
    if requested != _geometry or requested is None:
        raise ValueError("request and server target-KV geometry disagree")
    return {
        "target_kv_geometry": _geometry,
        "target_kv_state": "post_qk_norm_post_rope",
    }


class TargetKVCapture:
    def __init__(self, model, layer_ids):
        self.layer_ids = tuple(layer_ids)
        self.captured = {}
        self.handles = []
        self.serving = False
        for layer_id in self.layer_ids:
            attention = model.model.layers[layer_id].self_attn
            self.handles.append(
                attention.attn.register_forward_pre_hook(self._capture_layer(layer_id))
            )
        self.handles.append(model.logits_processor.register_forward_hook(self._output))

    def _capture_layer(self, layer_id):
        def capture(module, inputs):
            query, keys, values, batch = inputs[:4]
            if layer_id == min(self.layer_ids):
                self.captured.clear()
            if not batch.capture_hidden_mode.is_full():
                return
            if not batch.forward_mode.is_extend() and not (
                getattr(self, "serving", False)
                and batch.forward_mode.is_target_verify()
            ):
                raise ValueError("target-KV capture requires full prefill")
            if keys is None or values is None:
                raise ValueError(
                    "target-KV capture requires materialized post-RoPE K/V"
                )
            heads, head_dim = _geometry["heads"], _geometry["head_dim"]
            self.captured[layer_id] = torch.stack(
                (
                    keys.reshape(-1, heads, head_dim),
                    values.reshape(-1, heads, head_dim),
                ),
                dim=1,
            ).detach()

        return capture

    def _output(self, module, inputs, output):
        batch = inputs[3]
        try:
            if not batch.capture_hidden_mode.is_full():
                return
            if set(self.captured) != set(self.layer_ids):
                raise RuntimeError("incomplete target-KV layer capture")
            output.hidden_states = torch.stack(
                [self.captured[layer_id] for layer_id in self.layer_ids], dim=1
            )
            if getattr(self, "serving", False):
                output.hidden_states = output.hidden_states.flatten(1)
            output.last_hidden_states = inputs[1].detach()
        finally:
            self.captured.clear()


def install_target_kv_capture(model, server_args, *, layer_ids=None):
    global _geometry
    config = model.config
    if config.model_type != "qwen3":
        raise ValueError("online target-KV capture supports Qwen3 only")
    if server_args.tp_size != 1 or server_args.pp_size != 1:
        raise ValueError("target-KV capture requires TP=1 and PP=1")
    if not server_args.disable_cuda_graph or not server_args.disable_radix_cache:
        raise ValueError(
            "target-KV capture requires disabled CUDA graphs and radix cache"
        )
    if server_args.chunked_prefill_size != -1:
        raise ValueError("target-KV capture requires --chunked-prefill-size -1")
    if layer_ids is None:
        layer_ids = server_args.spec_capture_aux_layer_ids
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("target-KV capture requires explicit unique layer IDs")
    layer_types = (
        getattr(config, "layer_types", None)
        or ["full_attention"] * config.num_hidden_layers
    )
    if any(
        type(layer_id) is not int
        or not 0 <= layer_id < config.num_hidden_layers
        or layer_types[layer_id] != "full_attention"
        for layer_id in layer_ids
    ):
        raise ValueError("target-KV capture requires full-attention layer IDs")
    rope = (
        getattr(config, "rope_parameters", None)
        or getattr(config, "rope_scaling", None)
        or {}
    )
    if rope.get("rope_type", "default") != "default":
        raise ValueError("target-KV capture requires default NeoX RoPE")
    _geometry = {
        "layer_ids": list(layer_ids),
        "heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "rope_theta": rope.get("rope_theta", getattr(config, "rope_theta", None)),
        "rotary_dim": config.head_dim,
        "rope_layout": "neox",
    }
    model.target_kv_capture = TargetKVCapture(model, layer_ids)
