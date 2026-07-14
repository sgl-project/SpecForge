"""Keep the user-facing YAML reference synchronized with the typed schema."""

from pathlib import Path

from specforge.config import (
    Config,
    DataConfig,
    DeploymentConfig,
    DisaggregatedDeploymentConfig,
    ManagedLocalCaptureServerConfig,
    ManagedLocalMooncakeConfig,
    ManagedLocalStackConfig,
    ModelConfig,
    ProfilingConfig,
    RuntimeConfig,
    TrackingConfig,
    TrainerDeploymentConfig,
    TrainingConfig,
)


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "examples" / "configs" / "README.md"


def test_recipe_readme_names_every_typed_config_field():
    text = README.read_text(encoding="utf-8")
    schemas = (
        ("", Config),
        ("model", ModelConfig),
        ("data", DataConfig),
        ("training", TrainingConfig),
        ("tracking", TrackingConfig),
        ("profiling", ProfilingConfig),
        ("runtime", RuntimeConfig),
        ("deployment", DeploymentConfig),
        ("deployment.trainer", TrainerDeploymentConfig),
        ("deployment.disaggregated", DisaggregatedDeploymentConfig),
        ("deployment.disaggregated.managed_local", ManagedLocalStackConfig),
        (
            "deployment.disaggregated.managed_local.mooncake",
            ManagedLocalMooncakeConfig,
        ),
        (
            "deployment.disaggregated.managed_local.capture_servers[]",
            ManagedLocalCaptureServerConfig,
        ),
    )

    missing = []
    for prefix, schema in schemas:
        for field_name in schema.model_fields:
            path = f"{prefix}.{field_name}" if prefix else field_name
            if f"`{path}`" not in text:
                missing.append(path)

    assert not missing, f"examples/configs/README.md is missing config fields: {missing}"
