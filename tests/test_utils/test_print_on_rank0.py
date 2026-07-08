import importlib
import sys
import unittest
from types import ModuleType
from unittest import mock


def _load_utils_module():
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = object
    transformers_module.PreTrainedTokenizerFast = object

    transformers_utils_module = ModuleType("transformers.utils")
    transformers_utils_module.cached_file = mock.Mock(return_value=None)

    sys.modules.pop("specforge.utils", None)
    with mock.patch.dict(
        sys.modules,
        {
            "transformers": transformers_module,
            "transformers.utils": transformers_utils_module,
        },
    ):
        return importlib.import_module("specforge.utils")


utils = _load_utils_module()


class TestPrintOnRank0(unittest.TestCase):
    def test_logs_without_distributed_init(self):
        with (
            mock.patch.object(utils, "dist") as mock_dist,
            mock.patch.object(utils.logger, "info") as mock_info,
        ):
            mock_dist.is_available.return_value = False

            utils.print_on_rank0("hello")

        mock_info.assert_called_once_with("hello")

    def test_skips_nonzero_rank(self):
        with (
            mock.patch.object(utils, "dist") as mock_dist,
            mock.patch.object(utils.logger, "info") as mock_info,
        ):
            mock_dist.is_available.return_value = True
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 1

            utils.print_on_rank0("hello")

        mock_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()