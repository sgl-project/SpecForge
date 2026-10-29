"""Qwen3.5 native checkpoint contract, separate from the trainable network."""

from collections.abc import Mapping

from torch import Tensor


class Qwen3_5MTPAdapter:
    native_prefix = "mtp."
    target_embedding_keys = (
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    target_head_keys = (
        "model.language_model.lm_head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    )

    @classmethod
    def native_state_dict(cls, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Training and native serving use the same MTP tensor names."""
        return {
            key: value
            for key, value in state.items()
            if key.startswith(cls.native_prefix)
        }

    @classmethod
    def required_keys(
        cls, state: Mapping[str, Tensor], *, share_lm_head: bool
    ) -> set[str]:
        required = set(cls.native_state_dict(state))
        if share_lm_head:
            required.discard(f"{cls.native_prefix}lm_head.weight")
        return required

    @classmethod
    def allowed_extra_keys(cls, *, share_lm_head: bool) -> set[str]:
        # Merged target checkpoints may duplicate externally shared tensors.
        extra = {f"{cls.native_prefix}embed_tokens.weight"}
        if share_lm_head:
            extra.add(f"{cls.native_prefix}lm_head.weight")
        return extra

    @classmethod
    def patch_target_config(cls, base_config: dict, draft_config: dict) -> dict:
        """Supply the native MTP dimensions missing from some target configs.

        In particular, Qwen3.5 serving must not fall back to a different head_dim
        when constructing the MTP head. Preserve the existing in-place merge API.
        """
        target = base_config.get("text_config", base_config)
        source = draft_config.get("text_config", draft_config)
        for key in (
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            if key in source and target.get(key) != source[key]:
                print(
                    f"  overriding text_config.{key}: {target.get(key)} -> {source[key]}"
                )
                target[key] = source[key]
        return base_config
