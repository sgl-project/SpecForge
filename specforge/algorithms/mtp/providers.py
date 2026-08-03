"""Built-in MTP (Multi-Token Prediction) registration and executable providers.

MTP fine-tunes the single-layer draft head shipped natively with Qwen3.5-style
checkpoints.  The training signal is the target model's *final* (post-norm)
hidden state — unlike EAGLE3 it needs no aux-layer concat, so the offline
capture persists only ``input_ids`` / ``loss_mask`` / ``target_last_hidden_states``
(the streaming layout still carries the aux tensor the server patch always
produces; the strategy ignores it).

Draft construction initializes from the *native* ``mtp.*`` weights inside the
target checkpoint (fine-tuning) and shares + freezes the target embedding and
lm_head, mirroring the serving layout consumed by SGLang's
``Qwen3_5ForCausalLMMTP``.
"""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
    one_loss_token,
    online_needs_input_tools,
)
from specforge.algorithms.common.dflash_family_data import (
    MTP_NORMALIZER_ID,
    build_mtp_collator,
    build_mtp_offline_normalizer,
    build_mtp_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)

ALGORITHM_NAME = "mtp"
DRAFT_ARCHITECTURE = "Qwen3_5MTPDraftModel"

# MTP persists no aux-layer tensor, but the capture plan requires a non-empty
# layer list; a single layer keeps the (discarded) aux capture cheap.
_CAPTURE_LAYER_IDS = [1]


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import MTPTrainStrategy

    return MTPTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved MTP model and objective semantics."""

    mtp_config = getattr(draft_model.config, "mtp_config", None) or {}
    return {
        "mtp_draft_num_hidden_layers": int(
            getattr(draft_model.config, "num_hidden_layers", 1)
        ),
        "mtp_draft_vocab_size": int(getattr(draft_model.config, "vocab_size", 0)),
        "mtp_share_lm_head": bool(mtp_config.get("share_lm_head", True)),
        "mtp_attention_backend": str(
            getattr(draft_model.config, "_attn_implementation", "")
        ),
    }


def _init_from_native_mtp(cfg, draft_model) -> None:
    """Initialize the draft's ``mtp.*`` weights from the target checkpoint.

    Qwen3.5-style target checkpoints ship a native MTP head whose keys match
    the draft's flat ``mtp.*`` layout one-to-one.  Loading them turns training
    into fine-tuning of the native head (the only training mode for this
    algorithm); when no such weights exist we warn and leave the head at its
    random initialization.
    """

    import glob
    import os

    from safetensors import safe_open

    target_path = cfg.model.target_model_path
    native_mtp: dict = {}
    try:
        for shard in sorted(glob.glob(os.path.join(target_path, "*.safetensors"))):
            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    if key.startswith("mtp."):
                        native_mtp[key] = handle.get_tensor(key)
    except Exception as exc:  # pragma: no cover - depends on target checkpoint
        print(f"[mtp] could not scan native mtp.* weights in {target_path}: {exc}")

    if native_mtp:
        draft_model.load_state_dict(native_mtp, strict=False)
        print(
            f"[mtp] initialized {len(native_mtp)} native mtp.* weights from "
            f"{target_path} (native-MTP fine-tune)."
        )
    else:
        print(
            f"[mtp] WARNING: no native mtp.* weights found in {target_path}; "
            "the MTP head starts from random initialization."
        )


def _share_target_embeddings(cfg, draft_model, torch_dtype) -> None:
    """Share (and freeze) the target checkpoint's embed_tokens and lm_head."""

    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        device="cpu",
        dtype=torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    mtp_config = getattr(draft_model.config, "mtp_config", None) or {}
    share_lm_head = bool(mtp_config.get("share_lm_head", True))

    draft_model.embed_tokens.weight = target_components.embed_tokens.weight
    draft_model.embed_tokens.requires_grad_(False)
    if share_lm_head:
        draft_model.mtp.lm_head.weight = target_components.lm_head.weight
        draft_model.mtp.lm_head.requires_grad_(False)
    print(
        f"[mtp] shared target embed_tokens (and lm_head={share_lm_head}) with "
        "the MTP draft model (frozen)."
    )


def build_draft(cfg, draft_config):
    import torch

    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import warm_start_draft_model
    from specforge.utils import get_local_device

    torch_dtype = getattr(torch, cfg.model.torch_dtype)
    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(draft_config, torch_dtype=torch_dtype)

    _init_from_native_mtp(cfg, draft_model)
    _share_target_embeddings(cfg, draft_model, torch_dtype)

    if cfg.model.draft_checkpoint_path:
        warm_start_draft_model(
            draft_model,
            cfg.model.draft_checkpoint_path,
            draft_config=draft_config,
            strategy=cfg.training.strategy,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    return draft_model.to(device=get_local_device(), dtype=torch_dtype)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import AlgorithmModelParts
    from specforge.core.mtp import OnlineMTPModel

    return AlgorithmModelParts(
        model=OnlineMTPModel(draft_model=draft_model),
        capture_layers=None,
    )


def resolve_capture_layers(config, draft_config, target_config):
    return list(_CAPTURE_LAYER_IDS)


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


def algorithm_spec() -> AlgorithmSpec:
    ready = {
        "input_ids",
        "loss_mask",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures={DRAFT_ARCHITECTURE},
            default_architecture=DRAFT_ARCHITECTURE,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=MTP_NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                expected_auto_map_model="mtp.Qwen3_5MTPDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=one_loss_token,
            needs_input_tools=online_needs_input_tools,
            default_dataloader_num_workers=8,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=MTP_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature=None,
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_mtp_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_mtp_offline_normalizer,
                build_collator=build_mtp_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_mtp_collator,
            ),
        ),
    )


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
