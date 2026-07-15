"""Built-in chat templates stay independent of removed VLM integrations."""

import unittest

from specforge.data.template import TEMPLATE_REGISTRY


class TemplateRegistryTest(unittest.TestCase):
    def test_qwen2_vl_is_not_a_builtin_template(self):
        self.assertNotIn("qwen2-vl", TEMPLATE_REGISTRY.get_all_template_names())


if __name__ == "__main__":
    unittest.main()
