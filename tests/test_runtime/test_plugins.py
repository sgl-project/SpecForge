"""Contracts of the CLI platform plugin loader."""

import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from specforge import plugins


class LoadPluginsTest(unittest.TestCase):
    def test_retry_only_activates_unfinished_plugins(self):
        for function in (plugins.load_plugins, plugins._activate):
            function.cache_clear()
            self.addCleanup(function.cache_clear)

        first, second = mock.Mock(), mock.Mock()
        second.name = "tpu"
        second.load.return_value.side_effect = [ValueError("unavailable"), None]
        with mock.patch.object(plugins, "entry_points", return_value=[first, second]):
            with self.assertRaisesRegex(RuntimeError, "plugin 'tpu'") as error:
                plugins.load_plugins()
            self.assertIsInstance(error.exception.__cause__, ValueError)
            plugins.load_plugins()
            plugins.load_plugins()

        first.load.return_value.assert_called_once_with()
        self.assertEqual(second.load.return_value.call_count, 2)

    def test_cli_activates_plugins_before_importing_config(self):
        script = """
            import sys
            from unittest.mock import Mock, patch

            observed = []
            def activate():
                observed.append("specforge.config" in sys.modules)

            plugin = Mock()
            plugin.load.return_value = activate
            with patch("specforge.plugins.entry_points", return_value=[plugin]) as discover:
                import specforge.cli

            discover.assert_called_once_with(group="specforge.plugins")
            assert observed == [False], observed
            assert "specforge.config" in sys.modules
        """
        subprocess.run([sys.executable, "-c", textwrap.dedent(script)], check=True)
