"""Small inline draft settings for CPU configuration tests."""


def draft_settings(strategy="eagle3"):
    architectures = {
        "eagle3": "LlamaForCausalLMEagle3",
        "peagle": "PEagleDraftModel",
        "dflash": "DFlashDraftModel",
        "dflash2": "DFlash2DraftModel",
        "domino": "DominoDraftModel",
        "dspark": "DSparkDraftModel",
        "dspine": "DSpineDraftModel",
        "mtp": "Qwen3_5MTPDraftModel",
    }
    result = {
        "architectures": [architectures[strategy]],
        "vocab_size": 128,
        "draft_vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 256,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 5, 8]},
        "mask_token_id": 3,
    }
    if strategy not in ("eagle3", "peagle", "mtp"):
        result.update(
            block_size=4,
            num_target_layers=12,
            layer_types=["full_attention"],
            dflash_config={"target_layer_ids": [6], "mask_token_id": 3},
        )
    return result
