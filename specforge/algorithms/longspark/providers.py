"""LongSpark algorithm registration and providers."""

from specforge.algorithms.common.collation import pad_and_concatenate_features
from specforge.algorithms.common.defaults import no_missing_checkpoint_keys
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
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
)

from . import BLOCK_SIZE, FEATURES


def collate(features):
    return pad_and_concatenate_features(
        features,
        sequence_axes={key: 1 for key in FEATURES},
        required_keys=FEATURES,
    )


def build_collator(*_args, **_kwargs):
    return collate


def build_step(wrapped_model, *, seed=0, **_kwargs):
    from .model import LongSparkTrainStrategy

    return LongSparkTrainStrategy(wrapped_model, seed=seed)


def step_options(config):
    return {"seed": config.training.seed}


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def _draft_value(draft_config, key):
    if isinstance(draft_config, dict):
        return draft_config.get(key)
    return getattr(draft_config, key, None)


def capture_layers(config, draft_config, target_config):
    deployment = getattr(getattr(config, "deployment", None), "disaggregated", None)
    if deployment is not None and deployment.server_capture != "patch":
        raise ValueError("LongSpark K/V capture requires the patched SGLang backend")
    if getattr(target_config, "model_type", None) != "qwen3":
        raise ValueError("LongSpark supports Qwen3 target models only")
    target_head_dim = getattr(target_config, "head_dim", None) or (
        target_config.hidden_size // target_config.num_attention_heads
    )
    # The drafter reads target K/V and hidden states directly, so these
    # dimensions must match the target exactly.
    for key, target_value in (
        ("hidden_size", target_config.hidden_size),
        ("num_key_value_heads", target_config.num_key_value_heads),
        ("head_dim", target_head_dim),
        ("vocab_size", target_config.vocab_size),
    ):
        draft_value = _draft_value(draft_config, key)
        if draft_value != target_value:
            raise ValueError(
                f"LongSpark draft {key}={draft_value} must match the target "
                f"({target_value})"
            )
    if _draft_value(draft_config, "block_size") != BLOCK_SIZE:
        raise ValueError(f"LongSpark requires block_size={BLOCK_SIZE}")
    layers = list(_draft_value(draft_config, "target_layer_ids"))
    if not all(0 <= layer < target_config.num_hidden_layers for layer in layers):
        raise ValueError(
            f"target_layer_ids {layers} must be within "
            f"[0, {target_config.num_hidden_layers})"
        )
    return layers


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    import torch

    from specforge.algorithms.model_providers import AlgorithmModelParts
    from specforge.modeling.target.checkpoint import load_checkpoint_tensors
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.utils import get_local_device

    from .model import FrozenFinalNorm, LongSparkTrainingModel

    layers = capture_layers(config, draft_config, target_config)
    if (
        config.training.tp_size != 1
        or config.training.sp_ulysses_size != 1
        or config.training.sp_ring_size != 1
    ):
        raise ValueError("LongSpark training supports data parallelism only")
    device = get_local_device()
    dtype = getattr(torch, config.model.torch_dtype)
    target = TargetEmbeddingsAndHead.from_pretrained(
        config.model.target_model_path,
        embed_key=config.model.embedding_key,
        lm_head_key=config.model.lm_head_key,
        cache_dir=config.model.cache_dir,
        device=device.type,
        dtype=dtype,
    )
    state = load_checkpoint_tensors(
        config.model.target_model_path,
        keys=["model.norm.weight"],
        cache_dir=config.model.cache_dir,
    )
    norm = FrozenFinalNorm(state["model.norm.weight"], target_config.rms_norm_eps)
    model = LongSparkTrainingModel(
        draft_model,
        target.embed_tokens,
        target.lm_head,
        norm,
        num_anchors=config.training.num_anchors,
        ce_weight=config.training.dspark_ce_loss_alpha,
        l1_weight=config.training.dspark_l1_loss_alpha,
        gamma=config.training.loss_decay_gamma,
    ).to(device=device, dtype=dtype)
    return AlgorithmModelParts(model=model, capture_layers=layers)


def minimum_loss_tokens(_config, draft_config):
    return _draft_value(draft_config, "block_size")


def needs_input_tools(_config, _draft_model):
    return False


def eligible(loss_mask):
    from .sampling import has_trainable_block

    return has_trainable_block(loss_mask, BLOCK_SIZE)


def resume_contract(config, draft_model, training_model):
    draft_config = draft_model.config
    return {
        "longspark_target_layer_ids": tuple(draft_model.target_layer_ids),
        "longspark_block_size": int(draft_config.block_size),
        "longspark_global_memory_slots": int(draft_config.global_memory_slots),
        "longspark_recent_kv_window": int(draft_config.recent_kv_window),
        "longspark_num_anchors": int(training_model.num_anchors),
        "longspark_loss_decay_gamma": training_model.gamma,
        "longspark_ce_loss_alpha": float(training_model.ce_weight),
        "longspark_l1_loss_alpha": float(training_model.l1_weight),
    }


def create_registration():
    spec = AlgorithmSpec(
        name="longspark",
        draft=DraftRequirement(
            compatible_architectures={"LongSparkDraftModel"},
            default_architecture="LongSparkDraftModel",
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=FEATURES,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(attention_backends={"eager", "sdpa"}),
    )
    providers = AlgorithmProviders(
        algorithm_name="longspark",
        step=StepProvider(
            build=build_step,
            options=step_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(architecture="LongSparkDraftModel"),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=0,
            loss_mask_filter=eligible,
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="longspark",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                    extra_artifacts=(
                        ("target_keys", "target_keys"),
                        ("target_values", "target_values"),
                    ),
                ),
                build_collator=build_collator,
            ),
        ),
    )
    return make_registration(spec, providers)
