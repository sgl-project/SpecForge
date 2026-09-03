"""One resolver for ``model.sglang_*``: CLI flags for servers, kwargs for engines."""

import unittest

from specforge.config import Config
from specforge.launch_plan import _sglang_argv, resolve_sglang_engine_args


def _model(**sglang_fields):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
                **sglang_fields,
            },
            "data": {"prompts_path": "prompts.jsonl"},
            "training": {"strategy": "dflash", "max_steps": 1},
            "deployment": {"mode": "local_colocated"},
        }
    ).model


class SGLangEngineArgsTest(unittest.TestCase):
    def test_resolver_strips_the_prefix_applies_overrides_and_drops_none(self):
        model = _model(sglang_page_size=64, sglang_quantization=None)

        args = resolve_sglang_engine_args(
            model, overrides={"sglang_context_length": 4103, "sglang_dp_size": 1}
        )

        self.assertEqual(args["context_length"], 4103)
        self.assertEqual(args["dp_size"], 1)
        self.assertEqual(args["page_size"], 64)
        self.assertNotIn("quantization", args)
        self.assertFalse(any(name.startswith("sglang_") for name in args))

    def test_argv_renders_the_same_resolution_as_flags(self):
        model = _model(
            sglang_enable_torch_compile=True, sglang_disable_radix_cache=False
        )

        argv = _sglang_argv(model, overrides={"sglang_context_length": 64})

        self.assertIn("--enable-torch-compile", argv)
        self.assertNotIn("--disable-radix-cache", argv)
        self.assertNotIn("--enable-nccl-nvls", argv)
        self.assertEqual(argv[argv.index("--context-length") + 1], "64")


if __name__ == "__main__":
    unittest.main()
