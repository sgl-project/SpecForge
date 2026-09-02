"""Capture-layer hook resolution on the offline SGLang backend.

Runs where sglang is installed (CI); the backend module imports sglang.
"""

import types
import unittest
from unittest import mock

from specforge.offline_capture.sglang_backend.capture import OfflineSGLangCaptureBackend


def _backend_with(model):
    backend = object.__new__(OfflineSGLangCaptureBackend)
    backend.model_runner = types.SimpleNamespace(model=model)
    return backend


class CaptureLayerHookTest(unittest.TestCase):
    def test_dspark_prefers_its_native_capture_hook(self):
        model = mock.Mock(
            spec=["set_dspark_layers_to_capture", "set_dflash_layers_to_capture"]
        )

        _backend_with(model).set_capture_layers([1, 9, 17], capture_method="dspark")

        model.set_dspark_layers_to_capture.assert_called_once_with([1, 9, 17])
        model.set_dflash_layers_to_capture.assert_not_called()

    def test_dspark_falls_back_to_the_dense_dflash_hook(self):
        model = mock.Mock(spec=["set_dflash_layers_to_capture"])

        _backend_with(model).set_capture_layers([1, 9, 17], capture_method="dspark")

        model.set_dflash_layers_to_capture.assert_called_once_with([1, 9, 17])

    def test_missing_capture_hooks_fail_with_the_tried_names(self):
        model = mock.Mock(spec=[])

        with self.assertRaisesRegex(RuntimeError, "set_dspark_layers_to_capture"):
            _backend_with(model).set_capture_layers([1], capture_method="dspark")

    def test_unknown_capture_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "capture method"):
            _backend_with(mock.Mock()).set_capture_layers([1], capture_method="mtp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
