import re
import unittest

from specforge.data.parse import GeneralParser
from specforge.data.template import ChatTemplate


class TestAssistantHeaderAlternates(unittest.TestCase):

    def test_longest_header_matches_before_historical_alternate(self):
        template = ChatTemplate(
            assistant_header="<|im_start|>assistant\n<think>\n",
            assistant_header_alternates=["<|im_start|>assistant\n"],
            user_header="<|im_start|>user\n",
            end_of_turn_token="<|im_end|>\n",
        )
        parser = GeneralParser(tokenizer=object(), chat_template=template)
        rendered = (
            "<|im_start|>user\nfirst<|im_end|>\n"
            "<|im_start|>assistant\nhistorical<|im_end|>\n"
            "<|im_start|>user\nsecond<|im_end|>\n"
            "<|im_start|>assistant\n<think>\nreasoning\n</think>\n\n"
            "final<|im_end|>\n"
        )

        matches = list(re.finditer(parser.assistant_pattern, rendered, re.DOTALL))

        self.assertEqual(
            [match.group(1) for match in matches],
            [
                "historical<|im_end|>\n",
                "reasoning\n</think>\n\nfinal<|im_end|>\n",
            ],
        )


if __name__ == "__main__":
    unittest.main()
