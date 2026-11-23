"""Compose canonical models/objectives with existing data and trainer ports."""

import logging
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_config(cfg, *, target_config=None):
    from specforge.config.draft.codec import parse_draft

    draft = cfg.draft
    if isinstance(draft, str):
        import json

        import yaml

        text = Path(draft).read_text()
        try:
            draft = json.loads(text)
        except json.JSONDecodeError:
            draft = yaml.safe_load(text)
    supplied = (
        parse_draft(draft, target_config=target_config) if draft is not None else None
    )
    initialization = cfg.initialization
    source = initialization.checkpoint if initialization else None
    if cfg.resume_path:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.resume_path)
    if source:
        from specforge.loading import read_draft

        loaded = read_draft(
            source,
            target=cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            adapter=initialization.adapter if initialization else None,
        ).config
        if supplied is not None and supplied != loaded:
            raise ValueError(
                "draft must exactly match checkpoint structure; use weights_from to change it"
            )
        return loaded
    if supplied is None:
        raise ValueError("canonical training requires draft or a checkpoint")
    return supplied


def capture_layers(cfg, draft, target_config):
    from specforge.config.draft.codec import parse_draft

    if isinstance(draft, dict):
        draft = parse_draft(
            draft,
            target_config=target_config,
        )
    from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec

    features = draft.spec_config.target_features
    if isinstance(features, FinalHiddenFeatureSpec):
        if target_config["num_hidden_layers"] < 2:
            raise ValueError(
                "final-hidden capture requires a target with at least two layers"
            )
        # DFLASH capture requires an auxiliary tap; MTP discards this feature.
        return [0]
    return list(features.target_layer_ids)


def minimum_loss_tokens(cfg, draft, *, count):
    return count


def _objective_settings(cfg):
    return dict(
        name=cfg.training.objective,
        options=cfg.training.objective_options,
        training=cfg.training,
        loss=cfg.training.loss,
        weighting=cfg.training.weighting,
    )


def resolve_run(cfg, *, target_config=None):
    """Reuse deployment/data ports while resolving structure and loss separately."""
    from specforge.algorithms.common.providers import (
        AlgorithmProviders,
        DraftConfigProvider,
        ModelProvider,
        OfflineCaptureLayout,
        OfflineDataProvider,
        ServerCaptureLayout,
        ServerStreamingProvider,
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
    from specforge.algorithms.objectives import resolve_objective
    from specforge.config.draft.codec import encode_config

    from . import canonical_data

    draft = resolve_config(cfg, target_config=target_config)
    if cfg.initialization and cfg.initialization.from_target:
        from specforge.checkpoint.native import validate_target_initialization
        from specforge.checkpoint.resolve import read_target_config

        target = (
            target_config
            if target_config is not None
            else read_target_config(
                cfg.model.target_model_path, cache_dir=cfg.model.cache_dir
            )
        )
        validate_target_initialization(draft, target, cfg.initialization.from_target)
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    from specforge.algorithms.objectives.base import validate_anchor_sampling

    # At resolution, before a producer ingests data for a run that cannot train.
    validate_anchor_sampling(
        resolved.name,
        resolved.registration.training,
        cfg.training.anchor_sampling,
        cfg.training.seed,
    )
    if resolved.execution.teacher_chunk_size and cfg.mode != "offline":
        raise ValueError("compact teacher currently requires offline training")
    from specforge.config.schema import ModelConfig

    init = cfg.initialization
    scratch = not (init and (init.checkpoint or init.weights_from))
    used_keys = {
        "lm_head_key": "lm_head" in draft.target_bindings or resolved.requires_teacher,
        "embedding_key": "embed_tokens" in draft.target_bindings
        or resolved.requires_teacher
        or (scratch and cfg.model.load_target_embedding),
    }
    # on resume the recorded model provenance pins both keys instead
    for field, used in () if cfg.resume_path else used_keys.items():
        if (
            not used
            and getattr(cfg.model, field) != ModelConfig.model_fields[field].default
        ):
            raise ValueError(f"model.{field} is not used by this draft initialization")
    objective, entry = resolved.name, resolved.registration
    requirements = entry.training
    family = entry.runtime_name or objective
    if cfg.training.tp_size != 1:
        raise ValueError("canonical objectives require trainer TP size one")
    sequence_parallel = cfg.training.sp_ulysses_size * cfg.training.sp_ring_size > 1
    if sequence_parallel and (
        not requirements.supports_sequence_parallel
        or cfg.training.attention_backend != "usp"
    ):
        raise ValueError("sequence parallelism requires an objective supporting USP")
    if sequence_parallel:
        from specforge.losses.position_weights import DPACE_LOSS_TYPES

        if resolved.weighting.kind in DPACE_LOSS_TYPES:
            raise ValueError(
                "D-PACE weighting is not supported with sequence parallelism: "
                "chain weights are normalized within each sequence shard"
            )
    # Retain the established strategy name for data/rendezvous identifiers.
    options = asdict(resolved.objective)
    if cfg.training.weighting is not None:
        # An explicit weighting replaces these. build_bundle resolves this
        # config again and must not read the objective's defaults as user input.
        for name in getattr(resolved.objective, "default_weighting_options", ()):
            if name not in cfg.training.objective_options:
                del options[name]
    training = cfg.training.model_copy(
        update={"objective": objective, "objective_options": options}
    )
    cfg = cfg.model_copy(update={"draft": encode_config(draft), "training": training})
    keys = tuple(entry.inputs)
    normalizer = "canonical_unshifted_v1"
    storage = OfflineStorageContract(
        format="specforge_hidden_states_v1",
        normalizer=normalizer,
        required_tensors=frozenset(keys),
    )
    target_repr = "hidden_state" if "target_last_hidden_states" in keys else None
    representations = frozenset({target_repr}) if target_repr else frozenset()
    contracts = [
        FeatureContract(
            mode=FeatureMode.OFFLINE,
            modality="text",
            required_tensors=frozenset(keys),
            optional_tensors=frozenset({"attention_mask"}),
            allowed_target_representations=representations,
            default_target_representation=target_repr,
            storage=storage,
        )
    ]
    # Generic capture supplies aux/final target states. DMTP needs an external
    # frozen-MTP feature producer, so only precomputed records are accepted.
    layout = (
        None
        if requirements.capture_method is None
        else OfflineCaptureLayout(
            capture_method=requirements.capture_method,
            aux_feature="hidden_states" if "hidden_states" in keys else None,
            last_hidden_feature="target_last_hidden_states" if target_repr else None,
            passthrough=(("input_ids", "input_ids"), ("loss_mask", "loss_mask")),
        )
    )
    offline = OfflineDataProvider(
        modality="text",
        normalizer_id=normalizer,
        build_reader=partial(canonical_data.build_reader, strategy=family, keys=keys),
        build_normalizer=partial(
            canonical_data.build_normalizer,
            keys=keys,
            loss_mask_filter=requirements.loss_mask_filter,
            overlap=getattr(resolved.objective, "ttt_length", 1),
        ),
        build_collator=partial(canonical_data.build_collator, keys=keys),
        capture_layout=layout,
    )
    streaming = ()
    if layout is not None:
        server_layout = ServerCaptureLayout(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        )
        streaming = (
            ServerStreamingProvider(
                modality="text",
                capture_method=requirements.capture_method,
                target_representation=target_repr,
                layout=server_layout,
                build_collator=partial(canonical_data.build_collator, keys=keys),
                select_layout=None,
            ),
        )
        contracts.append(
            replace(contracts[0], mode=FeatureMode.STREAMING, storage=None)
        )
    vocab = requirements.supports_vocab_mapping
    if cfg.model.vocab_mapping_path and not vocab:
        raise ValueError("model.vocab_mapping_path requires a draft vocabulary")
    capabilities = AlgorithmCapabilities(
        attention_backends=requirements.attention_backends,
        required_batch_size=requirements.required_batch_size,
        supports_vocab_mapping=vocab,
        supports_trim_loss_positions=True,
        supports_compact_teacher=vocab,
        supports_liger_kernel=True,
    )
    spec = AlgorithmSpec(
        name=family,
        draft=DraftRequirement(
            compatible_architectures=frozenset({draft.architecture}),
            default_architecture=draft.architecture,
        ),
        feature_contracts=tuple(contracts),
        capabilities=capabilities,
    )
    provider = ModelProvider(
        build_bundle=build_bundle,
        default_dataloader_num_workers=requirements.dataloader_workers,
        loss_mask_filter=requirements.loss_mask_filter,
        draft_config=DraftConfigProvider(architecture=draft.architecture),
        resolve_capture_layers=capture_layers,
        minimum_loss_tokens=partial(
            minimum_loss_tokens, count=requirements.minimum_loss_tokens
        ),
    )
    providers = AlgorithmProviders(
        algorithm_name=family,
        model=provider,
        offline=(offline,),
        server_streaming=streaming,
        vocab_mapping_modes=spec.feature_modes if vocab else frozenset(),
    )
    return cfg, make_registration(spec, providers)


@dataclass(frozen=True)
class LigerChoice:
    """Resolved ``model.use_liger_kernel`` for one trainer platform."""

    enabled: bool
    reason: str

    def describe(self):
        return f"Liger kernels: {'on' if self.enabled else 'off'} ({self.reason})"


#: Liger kernels are Triton programs. SpecForge validates them on CUDA and ROCm;
#: Liger also ships Ascend and XPU backends that SpecForge has not validated.
_LIGER_VALIDATED_PLATFORMS = ("cuda", "rocm")
_LIGER_UNVALIDATED_PLATFORMS = ("npu", "xpu")


def trainer_platform():
    """This process's accelerator; PyTorch reports ROCm as ``cuda``."""
    import torch

    from specforge.utils import get_device_type

    device_type = get_device_type()
    if device_type == "cuda" and getattr(torch.version, "hip", None) is not None:
        return "rocm"
    return device_type


def liger_choice(requested, *, by_default, hidden_act, platform=None):
    """Resolve model.use_liger_kernel for a trainer platform (default: this one).

    An unset flag enables Liger only for objectives that declare it as their
    default, on a CUDA/ROCm trainer with a silu/swish draft. ``true`` never
    degrades silently: it raises where Liger cannot run.
    """
    platform = platform or trainer_platform()
    if requested is False:
        return LigerChoice(False, "model.use_liger_kernel=false")
    supported_activation = hidden_act in ("silu", "swish")
    if requested is None:
        if not by_default:
            return LigerChoice(False, "auto: not a default for this objective")
        if platform not in _LIGER_VALIDATED_PLATFORMS:
            return LigerChoice(
                False, f"auto: trainer device {platform!r} is not CUDA/ROCm"
            )
        if not supported_activation:
            return LigerChoice(
                False, f"auto: draft hidden_act {hidden_act!r} is not silu/swish"
            )
        return LigerChoice(True, f"auto: {platform} trainer")
    if platform not in _LIGER_VALIDATED_PLATFORMS + _LIGER_UNVALIDATED_PLATFORMS:
        raise ValueError(
            "model.use_liger_kernel=true requires a Triton GPU backend: CUDA or "
            "ROCm, or Ascend NPU/XPU where it is unvalidated in SpecForge; this "
            f"trainer's device is {platform!r}. Remove the key or set it to "
            "false to use the PyTorch RMSNorm/SwiGLU."
        )
    if not supported_activation:
        raise ValueError(
            "model.use_liger_kernel=true requires the draft hidden_act to be "
            f"'silu' or 'swish' (Liger SwiGLU); got {hidden_act!r}. Remove the "
            "key or set it to false to use the PyTorch MLP."
        )
    if platform in _LIGER_UNVALIDATED_PLATFORMS:
        return LigerChoice(True, f"explicit, unvalidated in SpecForge on {platform}")
    return LigerChoice(True, "explicit")


def preflight_kernels(cfg):
    """Import the Liger kernels this run's draft build will resolve.

    Called before services start or a consumer waits on its producer, so a
    missing install fails at startup. build_bundle still owns the resolution.
    """
    requested = cfg.model.use_liger_kernel
    if requested is False:
        return
    if requested is None:
        from specforge.algorithms.objectives import resolve_objective
        from specforge.config.draft.mlp.dense import DenseFFNSpec

        if trainer_platform() not in _LIGER_VALIDATED_PLATFORMS:
            return  # auto never enables Liger here, so there is nothing to read
        draft = resolve_config(cfg)
        acts = {
            node.hidden_act
            for node in draft.mlp_configs.values()
            if isinstance(node, DenseFFNSpec)
        }
        resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
        choice = liger_choice(
            None,
            by_default=resolved.registration.training.liger_by_default,
            hidden_act="silu" if acts <= {"silu", "swish"} and acts else "other",
        )
        if not choice.enabled:
            return
    from specforge.modeling.draft.layers.kernels import load_liger_kernels

    load_liger_kernels()


def _raise_flex_recompile_limit():
    """Let compiled FlexAttention follow changing block-mask shapes.

    Past the limit torch stops compiling the frame and runs it eagerly without
    an error. The limit is process-wide; 64 is the value the block objectives
    were tuned with.
    """
    import torch._dynamo as dynamo

    dynamo.config.recompile_limit = max(dynamo.config.recompile_limit, 64)


def build_bundle(cfg, algorithm, topology):
    import torch

    from specforge.algorithms.common.providers import (
        MODEL_PROVENANCE_CONTRACT_KEY,
        StepRuntimeConfig,
    )
    from specforge.algorithms.objectives import (
        build_objective_model,
        objective_record,
        resolve_objective,
    )
    from specforge.checkpoint.target import TargetWeights
    from specforge.config.draft.codec import encode_config
    from specforge.config.draft.codec.target import (
        bound_vocab_size,
        resolve_target_logits,
    )
    from specforge.config.schema import ModelConfig
    from specforge.loading import initialize_draft, load_draft
    from specforge.modeling.draft.bindings import BoundLMHead
    from specforge.modeling.draft.layers.kernels import (
        configure_kernels,
        liger_hidden_act,
    )
    from specforge.training.assembly import ModelBundle, _device
    from specforge.training.model_loading import training_target_config

    target = training_target_config(cfg)
    draft = resolve_config(cfg, target_config=target)
    resolved = resolve_objective(draft.architecture, **_objective_settings(cfg))
    dtype, device = getattr(torch, cfg.model.torch_dtype), _device()
    init = cfg.initialization
    if (
        resolved.registration.training.explicit_initialization
        and init is None
        and not cfg.resume_path
    ):
        raise ValueError(
            f"{resolved.name} requires an explicit initialization source; "
            "set initialization: {} to train from scratch, or use resume"
        )
    source = init.checkpoint if init else None
    if cfg.resume_path:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.resume_path)
    target_weights = TargetWeights(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        keys={
            name: getattr(cfg.model, field)
            for name, field in (
                ("embed_tokens", "embedding_key"),
                ("lm_head", "lm_head_key"),
            )
            if getattr(cfg.model, field) != ModelConfig.model_fields[field].default
        },
    )
    common = dict(
        target=target_weights,
        draft=draft,
        device=device,
        dtype=dtype,
        backend=(
            "eager"
            if cfg.training.attention_backend == "usp"
            else cfg.training.attention_backend
        ),
        cache_dir=cfg.model.cache_dir,
    )
    if source:
        model = load_draft(
            source,
            **common,
            adapter=init.adapter if init else None,
        ).train()
    else:
        weights = init.weights_from if init else None
        from_target = dict(init.from_target or {}) if init else {}
        cast_copies = ()
        # an explicit copy of the embedding, or of its tied alias, takes precedence
        explicit = ("", "embed_tokens", "embed_tokens.weight") + (
            ("lm_head", "lm_head.weight") if draft.tie_word_embeddings else ()
        )
        if (
            weights is None
            and cfg.model.load_target_embedding
            and "embed_tokens" not in draft.target_bindings
            and not any(k in from_target for k in explicit)
        ):
            from_target["embed_tokens.weight"] = target_weights.resolve(
                "embed_tokens"
            ).key
            # legacy load_embedding parity: follow model.torch_dtype
            cast_copies = ("embed_tokens.weight",)
        model = initialize_draft(
            **common,
            weights_from=weights.checkpoint if weights else None,
            mapping=weights.mapping if weights else None,
            allow_new=weights.allow_new if weights else (),
            require_loaded=tuple(cfg.training.frozen) + resolved.frozen_parameters,
            from_target=from_target or None,
            cast_target_copies=cast_copies,
        )
    teacher = None
    if resolved.requires_teacher:
        transform = resolve_target_logits(target_weights.config)
        if isinstance(model.lm_head, BoundLMHead) and model.lm_head.config == transform:
            teacher = model.lm_head
        else:
            teacher = BoundLMHead(
                transform,
                vocab_size=bound_vocab_size(target),
                hidden_size=target["hidden_size"],
                device=device,
                dtype=dtype,
            )
            teacher.load_target_weight(target_weights.resolve("lm_head").read())
    if teacher is not None and hasattr(model, "vocabulary"):
        width = bound_vocab_size(target)
        if model.vocabulary.vocab_size != width:
            # t2d must cover every teacher logit column
            raise ValueError(
                f"draft vocab_size {model.vocabulary.vocab_size} must equal the "
                f"target logit width {width} (padded_vocab_size when the target pads)"
            )
    liger = liger_choice(
        cfg.model.use_liger_kernel,
        by_default=resolved.registration.training.liger_by_default,
        hidden_act=liger_hidden_act(model),
    )
    logger.info(liger.describe())
    use_liger = liger.enabled
    use_fused_conv = configure_kernels(
        model, use_liger=use_liger, use_fused_conv=cfg.model.use_fused_conv
    )
    if cfg.training.attention_backend == "flex_attention":
        _raise_flex_recompile_limit()
    if hasattr(model, "vocabulary") and cfg.model.vocab_mapping_path:
        mapping = torch.load(
            cfg.model.vocab_mapping_path, map_location=device, weights_only=True
        )
        # a loaded or copied vocabulary was trained with its own mapping
        if bool(model.vocabulary.ready) and (
            not torch.equal(model.vocabulary.t2d, mapping["t2d"])
            or not torch.equal(model.vocabulary.d2t, mapping["d2t"])
        ):
            raise ValueError(
                "vocab_mapping_path disagrees with the loaded draft vocabulary"
            )
        model.vocabulary.load_mapping(t2d=mapping["t2d"], d2t=mapping["d2t"])
    sequence_parallel = None
    if cfg.training.attention_backend == "usp":
        from .sequence_parallel import sequence_parallel_groups

        sequence_parallel = sequence_parallel_groups(topology)
    runtime = build_objective_model(
        model,
        **_objective_settings(cfg),
        frozen=cfg.training.frozen,
        target_head=teacher,
        frozen_modules=[model.get_submodule(name) for name in draft.target_bindings]
        + ([teacher] if teacher is not None else []),
        attention_backend=cfg.training.attention_backend,
        sequence_parallel=sequence_parallel,
        anchor_sampling=cfg.training.anchor_sampling,
        anchor_seed=cfg.training.seed,
    )
    runtime.fsdp_param_dtype = dtype
    contract = {
        "canonical_draft_config": encode_config(draft),
        "canonical_objective": runtime.objective_name,
        "canonical_objective_options": objective_record(
            runtime.objective_name, asdict(runtime.objective)
        ),
        "canonical_use_liger_kernel": use_liger,
        "canonical_use_fused_conv": use_fused_conv,
        # A resume keeps drawing anchors as the run did: the host draw keys on
        # the seed, so it is part of the record.
        "canonical_anchor_sampling": (
            ("host", cfg.training.seed)
            if cfg.training.anchor_sampling == "host"
            else "device"
        ),
        "canonical_frozen": tuple(sorted(set(cfg.training.frozen))),
        "canonical_loss": runtime.loss_configuration.to_dict(),
        "canonical_weighting": asdict(runtime.weighting),
        "canonical_loss_execution": asdict(runtime.execution),
        "canonical_loss_implementations": runtime.loss_implementations.names,
    }
    from specforge.training.provenance import model_resume_provenance

    provenance = model_resume_provenance(
        cfg, draft, target, capture_layers=capture_layers(cfg, draft, target)
    )
    contract[MODEL_PROVENANCE_CONTRACT_KEY] = tuple(sorted(provenance.items()))
    return ModelBundle(
        model=runtime,
        draft_model=model,
        draft_config=draft,
        target_head=teacher,
        target_hidden_size=target["hidden_size"],
        target_vocab_size=bound_vocab_size(target),
        draft_vocab_size=draft.draft_vocab_size,
        capture_layers=capture_layers(cfg, draft, target),
        strategy_kwargs=StepRuntimeConfig(
            options={
                "name": algorithm.name,
                "objective": runtime.objective_name,
                "required_features": tuple(runtime.registration.inputs),
            },
            resume_contract=contract,
            allowed_missing_checkpoint_keys=frozenset(),
        ),
    )
