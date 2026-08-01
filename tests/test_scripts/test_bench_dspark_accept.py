import unittest
from types import SimpleNamespace
from unittest import mock

from scripts import bench_dspark_accept


class DSparkAcceptBenchmarkTest(unittest.TestCase):
    def test_request_uses_sampling_protocol_and_normalizes_url(self):
        args = SimpleNamespace(
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=128,
            stop_token_ids=[200006, 200000],
            timeout_s=30,
        )
        response = mock.Mock()
        response.json.return_value = [{"meta_info": {"spec_accept_length": 3.5}}]

        with mock.patch.object(
            bench_dspark_accept.requests,
            "post",
            return_value=response,
        ) as post:
            result = bench_dspark_accept.send_one(
                "http://127.0.0.1:30000/",
                "prompt",
                args,
            )

        response.raise_for_status.assert_called_once_with()
        post.assert_called_once_with(
            "http://127.0.0.1:30000/generate",
            json={
                "text": "prompt",
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": -1,
                    "max_new_tokens": 128,
                    "stop_token_ids": [200006, 200000],
                },
            },
            timeout=30,
        )
        self.assertEqual(result["meta_info"]["spec_accept_length"], 3.5)

    def test_chat_template_falls_back_without_reasoning_effort(self):
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.side_effect = [TypeError, "rendered"]
        messages = [{"role": "user", "content": "question"}]

        result = bench_dspark_accept.apply_chat_template(
            tokenizer,
            messages,
            reasoning_effort=0.99,
        )

        self.assertEqual(result, "rendered")
        self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
        _, first_kwargs = tokenizer.apply_chat_template.call_args_list[0]
        _, second_kwargs = tokenizer.apply_chat_template.call_args_list[1]
        self.assertEqual(first_kwargs["reasoning_effort"], 0.99)
        self.assertNotIn("reasoning_effort", second_kwargs)


if __name__ == "__main__":
    unittest.main()
