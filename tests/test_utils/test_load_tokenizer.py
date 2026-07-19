import unittest
from unittest import mock

import specforge.utils as utils


class LoadTokenizerFallbackTest(unittest.TestCase):
    def test_unknown_model_config_falls_back_to_generic_fast_tokenizer(self):
        for error_type in (AttributeError, ValueError):
            with self.subTest(error_type=error_type.__name__):
                sentinel = object()
                with (
                    mock.patch.object(
                        utils.AutoTokenizer,
                        "from_pretrained",
                        side_effect=error_type("unknown remote model config"),
                    ) as auto_loader,
                    mock.patch.object(
                        utils.PreTrainedTokenizerFast,
                        "from_pretrained",
                        return_value=sentinel,
                    ) as fast_loader,
                ):
                    result = utils.load_tokenizer(
                        "org/model",
                        trust_remote_code=True,
                        cache_dir="/tmp/model-cache",
                    )

                self.assertIs(result, sentinel)
                auto_loader.assert_called_once_with(
                    "org/model",
                    trust_remote_code=True,
                    cache_dir="/tmp/model-cache",
                )
                fast_loader.assert_called_once_with(
                    "org/model",
                    cache_dir="/tmp/model-cache",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
