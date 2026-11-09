"""Resolved canonical capture contracts for runtime tests."""

from dataclasses import replace
from pathlib import Path

from specforge.algorithms.registry import AlgorithmRegistry
from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.training.canonical import resolve_run
from tests.test_config.test_draft_mtp_spec import mtp_config

PRESETS = Path(__file__).resolve().parents[2] / "specforge/config/draft/presets"


def builtin_algorithm_registry(*, teacher_metrics=False):
    registrations = []
    for name in ("eagle3", "peagle", "dflash", "domino", "dspark", "dspine", "mtp"):
        if name == "mtp":
            draft = encode_config(mtp_config())
        elif name == "dspine":
            from specforge.config.draft.codec.legacy import decode_legacy_config
            from tests.test_config.test_draft_legacy_dflash import target_for
            from tests.test_config.test_draft_legacy_dspine import dspine_payload

            payload = dspine_payload()
            draft = encode_config(
                decode_legacy_config(payload, target_config=target_for(payload))
            )
        else:
            draft = str(PRESETS / f"qwen3-8b-{name}.yaml")
        cfg = Config.model_validate(
            dict(
                draft=draft,
                model=dict(target_model_path="target"),
                data=dict(hidden_states_path="features"),
                training=dict(
                    objective_options=(
                        {"mask_token_id": 0}
                        if name == "peagle"
                        else (
                            {"teacher_metrics": teacher_metrics}
                            if name == "dflash"
                            else {}
                        )
                    )
                ),
            )
        )
        registration = resolve_run(cfg)[1]
        from tests.test_runtime._fixtures import build_strategy

        registration = replace(
            registration,
            providers=replace(
                registration.providers,
                step=replace(registration.providers.step, build=build_strategy),
            ),
        )
        registrations.append(registration)
    return AlgorithmRegistry(registrations)
