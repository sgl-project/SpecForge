"""Checked-in canonical JSON pins the on-disk format of every registered architecture."""

import dataclasses
import json
import unittest
from pathlib import Path

from specforge.config.draft.architecture import DFlash2Spec
from specforge.config.draft.codec import decode_config, encode_config
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.registry import _SPECS
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_domino_spec import domino_spec
from tests.test_config.test_draft_dspark_spec import dspark_spec
from tests.test_config.test_draft_dspine_spec import dspine_config
from tests.test_config.test_draft_eagle3_spec import eagle_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_config.test_draft_peagle_spec import peagle_config

FIXTURES = Path(__file__).parent / "fixtures" / "canonical"


def dflash2_config():
    base = draft_config()
    return dataclasses.replace(
        base,
        architecture="DFlash2DraftModel",
        spec_config=DFlash2Spec(
            block=base.spec_config.block,
            target_features=base.spec_config.target_features,
            selector_rank=8,
            selector_top_k=4,
            output_multiplier=1.0,
            final_logit_softcapping=None,
        ),
        conv_configs={"conv": GroupedCausalConvSpec(kernel_size=3, group_size=8)},
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense", sublayer_conv="conv")
        },
    )


EXPECTED = {
    "MTPDraftModel": mtp_config,
    "PEagleDraftModel": peagle_config,
    "Eagle3DraftModel": eagle_config,
    "DominoDraftModel": lambda: draft_config(
        architecture="DominoDraftModel", spec_config=domino_spec()
    ),
    "DFlashDraftModel": draft_config,
    "DFlash2DraftModel": dflash2_config,
    "DSparkDraftModel": lambda: draft_config(
        architecture="DSparkDraftModel", spec_config=dspark_spec()
    ),
    "DSpineDraftModel": dspine_config,
}


class CanonicalFixturesTest(unittest.TestCase):
    def test_every_registered_architecture_has_a_golden_fixture(self):
        self.assertEqual(set(EXPECTED), set(_SPECS))

    def test_golden_json_decodes_to_the_expected_config_and_back(self):
        for name, expected in EXPECTED.items():
            with self.subTest(architecture=name):
                payload = json.loads((FIXTURES / f"{name}.json").read_text())
                self.assertEqual(decode_config(payload), expected())
                self.assertEqual(encode_config(decode_config(payload)), payload)


if __name__ == "__main__":
    unittest.main()
