# coding=utf-8
"""Contracts of the specforge.plugins entry-point loader."""

from __future__ import annotations

import unittest
from unittest import mock

from specforge import plugins


class LoadPluginsTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(plugins, "_loaded", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_plugin_runs_once_per_process(self):
        register = mock.Mock()
        entry_point = mock.Mock(load=mock.Mock(return_value=register))
        with mock.patch.object(
            plugins, "entry_points", return_value=[entry_point]
        ) as found:
            plugins.load_plugins()
            plugins.load_plugins()
        found.assert_called_once_with(group=plugins.PLUGIN_GROUP)
        register.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
