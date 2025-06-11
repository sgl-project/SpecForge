import json
import os

import torch
from safetensors.torch import load_file, save_file


def extract_draft_model(checkpoint_dir, output_dir):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    param_mapping = {
        "draft_decoder.layers.0.attn.k_proj.weight": "midlayer.self_attn.k_proj.weight",
        "draft_decoder.layers.0.attn.output_proj.weight": "midlayer.self_attn.o_proj.weight",
        "draft_decoder.layers.0.attn.q_proj.weight": "midlayer.self_attn.q_proj.weight",
        "draft_decoder.layers.0.attn.v_proj.weight": "midlayer.self_attn.v_proj.weight",
        "draft_decoder.layers.0.mlp.w1.weight": "midlayer.mlp.gate_proj.weight",
        "draft_decoder.layers.0.mlp.w2.weight": "midlayer.mlp.down_proj.weight",
        "draft_decoder.layers.0.mlp.w3.weight": "midlayer.mlp.up_proj.weight",
        "draft_decoder.layers.0.mlp_norm.scale": "midlayer.post_attention_layernorm.weight",
        "draft_decoder.norm.scale": "norm.weight",
        "feature_fusion.bias": "fc.bias",
        "feature_fusion.weight": "fc.weight",
        "input_embeds_norm.scale": "midlayer.hidden_norm.weight",
        "fused_features_norm.scale": "midlayer.input_layernorm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    draft_state_dict = {}
    for weight_map in index["weight_map"].items():
        weight_name, shard_name = weight_map
        if weight_name.startswith("draft.") or "lm_head" in weight_name:
            shard_path = os.path.join(checkpoint_dir, shard_name)
            shard_weights = load_file(shard_path)
            weight = shard_weights[weight_name]

            if weight_name.startswith("draft."):
                new_key = weight_name[6:]
            elif "language_model.lm_head" in weight_name:
                new_key = weight_name.replace("language_model.", "")
            else:
                new_key = weight_name

            if new_key in param_mapping:
                new_key = param_mapping[new_key]

            draft_state_dict[new_key] = weight

    if "feature_fusion.weight" not in draft_state_dict:
        draft_state_dict["feature_fusion.weight"] = torch.nn.init.xavier_uniform_(
            torch.empty(5120, 5120 * 3)
        )
        draft_state_dict["feature_fusion.bias"] = torch.zeros(5120)

    if "midlayer.input_layernorm.weight" not in draft_state_dict:
        draft_state_dict["midlayer.input_layernorm.weight"] = torch.ones(5120)

    if "midlayer.hidden_norm.weight" not in draft_state_dict:
        draft_state_dict["midlayer.hidden_norm.weight"] = torch.ones(5120)

    os.makedirs(output_dir, exist_ok=True)
    save_file(draft_state_dict, os.path.join(output_dir, "model.safetensors"))

    config = {
        "architectures": ["LlamaForCausalLM"],
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 23, 44],
            "use_aux_hidden_state": True,
            "use_input_layernorm_in_first_layer": True,
            "use_last_layernorm": True,
            "use_mtp_layernorm": False,
        },
        "attention_bias": True,
        "model_type": "llama",
        "vocab_size": 202_048,
        "hidden_size": 5120,
        "num_hidden_layers": 1,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 32768,
        "max_position_embeddings": 10485760,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    tokenizer_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
    ]

    for file in tokenizer_files:
        src_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(src_path):
            with open(src_path, "rb") as fsrc:
                with open(os.path.join(output_dir, file), "wb") as fdst:
                    fdst.write(fsrc.read())

    print(f"Draft model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save draft model",
    )
    args = parser.parse_args()
    extract_draft_model(args.checkpoint_dir, args.output_dir)
