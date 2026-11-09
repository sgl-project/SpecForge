"""Resolve draft sources and assemble training modules; checkpoint owns file I/O."""

from dataclasses import asdict
from pathlib import Path


def resolve_draft_config(cfg, *, target_config=None):
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
    if cfg.training.resume_from:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.training.resume_from)
    if source:
        from specforge.loading import read_draft

        loaded = read_draft(
            source,
            target=cfg.model.target_model_path,
            cache_dir=cfg.model.cache_dir,
            adapter=initialization.adapter if initialization else None,
            legacy_config=initialization.legacy_config if initialization else None,
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
    from specforge.modeling.target.target_utils import target_text_config

    if isinstance(draft, dict):
        draft = parse_draft(
            draft, target_config=target_text_config(target_config).to_dict()
        )
    features = draft.spec_config.target_features
    if hasattr(features, "target_layer_ids"):
        return list(features.target_layer_ids)
    target = getattr(target_config, "text_config", target_config)
    return [target.num_hidden_layers - 1]


def build_bundle(cfg, algorithm):
    import torch

    from specforge.algorithms.common.providers import (
        MODEL_PROVENANCE_CONTRACT_KEY,
        StepRuntimeConfig,
    )
    from specforge.algorithms.objectives import build_objective_model
    from specforge.config.draft.codec import encode_config
    from specforge.config.draft.codec.target import resolve_target_logits
    from specforge.loading import initialize_draft, load_draft
    from specforge.modeling.draft.bindings import BoundLMHead
    from specforge.modeling.target.target_head import TargetHead
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_hidden_size,
        target_text_config,
        target_vocab_size,
    )
    from specforge.training.assembly import ModelBundle, _device

    target = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    draft = resolve_draft_config(
        cfg, target_config=target_text_config(target).to_dict()
    )
    dtype, device = getattr(torch, cfg.model.torch_dtype), _device()
    init = cfg.initialization
    source = init.checkpoint if init else None
    if cfg.training.resume_from:
        from specforge.training.checkpoint import CheckpointManager

        source = CheckpointManager.resolve_resume_dir(cfg.training.resume_from)
    common = dict(
        target=cfg.model.target_model_path,
        draft=draft,
        device=device,
        dtype=dtype,
        backend=(
            "eager"
            if cfg.training.attention_backend in ("fa", "usp")
            else cfg.training.attention_backend
        ),
        cache_dir=cfg.model.cache_dir,
    )
    if source:
        model = load_draft(
            source,
            **common,
            adapter=init.adapter if init else None,
            legacy_config=init.legacy_config if init else None,
        ).train()
    else:
        weights = init.weights_from if init else None
        model = initialize_draft(
            **common,
            weights_from=weights.checkpoint if weights else None,
            mapping=weights.mapping if weights else None,
            allow_new=weights.allow_new if weights else (),
            from_target=init.from_target if init else None,
            legacy_config=init.legacy_config if init else None,
        )
    teacher = None
    needs_teacher = cfg.training.objective in (
        "eagle3",
        "peagle",
        "dspark",
        "dspine",
    ) or (
        cfg.training.objective in ("dflash", "dflash2")
        and cfg.training.objective_options.get("teacher_metrics", True)
    )
    if needs_teacher and "lm_head" in draft.target_bindings:
        teacher = model.lm_head
    elif needs_teacher:
        loaded = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            embedding_key=cfg.model.embedding_key,
            dtype=dtype,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        teacher = BoundLMHead(
            resolve_target_logits(target_text_config(target).to_dict()),
            vocab_size=target_vocab_size(target),
            hidden_size=target_hidden_size(target),
            device=device,
            dtype=dtype,
        )
        teacher.load_target_weight(loaded.fc.weight)
    if hasattr(model, "initialize_transfer_space") and not bool(model.transfer_ready):
        if source:
            raise ValueError("DSpine checkpoint has no initialized transfer space")
        model.initialize_transfer_space(teacher.bound_weight())
    if hasattr(model, "vocabulary") and cfg.model.vocab_mapping_path:
        mapping = torch.load(
            cfg.model.vocab_mapping_path, map_location=device, weights_only=True
        )
        if (
            source
            and bool(model.vocabulary.ready)
            and (
                not torch.equal(model.vocabulary.t2d, mapping["t2d"])
                or not torch.equal(model.vocabulary.d2t, mapping["d2t"])
            )
        ):
            raise ValueError("vocab_mapping_path disagrees with checkpoint vocabulary")
        model.vocabulary.load_mapping(t2d=mapping["t2d"], d2t=mapping["d2t"])
    runtime = build_objective_model(
        model,
        name=cfg.training.objective,
        options=cfg.training.objective_options,
        frozen=cfg.training.frozen,
        target_head=teacher,
    )
    from .kernels import configure_liger

    configure_liger(
        model,
        cfg.model.use_liger_kernel,
        default_enabled=cfg.training.objective in ("dflash", "dflash2"),
    )
    runtime.attention_backend = cfg.training.attention_backend
    if runtime.attention_backend == "usp":
        from .sequence_parallel import configure_sequence_parallel

        configure_sequence_parallel(runtime)
    runtime.frozen_target_modules = tuple(
        dict.fromkeys(
            [model.get_submodule(name) for name in draft.target_bindings]
            + ([teacher] if teacher is not None else [])
        )
    )
    runtime.fsdp_param_dtype = dtype
    contract = {
        "canonical_draft_config": encode_config(draft),
        "canonical_objective": runtime.objective_name,
        "canonical_objective_options": asdict(runtime.objective),
        "canonical_frozen": tuple(cfg.training.frozen),
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
        target_hidden_size=target_hidden_size(target),
        target_vocab_size=target_vocab_size(target),
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


def preflight_draft_kernels(cfg, *, algorithm):
    from specforge.training.assembly import _device
    from specforge.training.kernels import resolve_liger

    draft = resolve_draft_config(cfg)
    resolve_liger(
        cfg.model.use_liger_kernel,
        activations=[node.hidden_act for node in draft.mlp_configs.values()],
        device=_device().type,
        default_enabled=draft.architecture in ("DFlashDraftModel", "DFlash2DraftModel"),
    )


def training_target_config(cfg):
    """Load text metadata before parsing a training draft."""
    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
    )

    target = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return target_text_config(target).to_dict()
