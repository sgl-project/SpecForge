"""Check an HF export in isolated Python without importing installed SpecForge."""

import importlib
import importlib.abc
import sys
from pathlib import Path

import torch


class BlockSpecForgeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "specforge" or fullname.startswith("specforge."):
            raise ImportError("SpecForge must not be imported")


def main(export_dir):
    sys.meta_path.insert(0, BlockSpecForgeImports())
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(export_dir, trust_remote_code=True)
    assert model.config.to_dict()["architectures"] == ["DFlashDraftModel"]
    assert not model.is_backend_compatible()
    assert model.config.draft == config.draft
    assert config.specforge_artifact == {"format": "specforge", "version": 1}
    if not model.config.draft.target_bindings:
        package = type(model).__module__.rsplit(".", 1)[0]
        LayerInputs = importlib.import_module(
            package + ".specforge_modeling_draft_layers_inputs"
        ).LayerInputs
        Layout = importlib.import_module(
            package + ".specforge_modeling_draft_layers_attention_masks"
        ).AttentionLayout
        batch = torch.load(export_dir.parent / "expected.pt", weights_only=True)
        layout = Layout(
            query_positions=batch["q"],
            key_positions=batch["k"],
            visibility=batch["visibility"],
        )
        entries = [
            LayerInputs(attention_layout=layout, position_ids=batch["k"])
        ] * model.config.draft.num_hidden_layers
        torch.testing.assert_close(
            model(batch["ids"], batch["features"], entries).logits,
            batch["logits"],
            rtol=0,
            atol=0,
        )
    resaved = export_dir.with_name(export_dir.name + "_resaved")
    model.save_pretrained(resaved)
    again = AutoModel.from_pretrained(resaved, trust_remote_code=True)
    assert again.config.to_dict()["architectures"] == ["DFlashDraftModel"]
    assert again.config.specforge_artifact == config.specforge_artifact


if __name__ == "__main__":
    main(Path(sys.argv[1]))
