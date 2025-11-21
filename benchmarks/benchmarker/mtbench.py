"""
MT-Bench benchmark evaluation script.
Adapted from https://github.com/chromecast56/sglang/blob/6f145d2eadb93a116134f703358ce76f15381045/benchmark/mtbench/bench_sglang.py
"""

from typing import Any, Dict, List, Optional, Tuple

from sglang.utils import download_and_cache_file, read_jsonl

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_multi_turn_sgl_function

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."


@BENCHMARKS.register("mtbench")
class MTBenchBenchmarker(Benchmarker):
    """MT-Bench benchmark implementation."""

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        # support categorical data for mtbench
        if subset is None:
            subset = ["all"]
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        """Load and preprocess MT-Bench dataset."""
        url = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
        download_and_cache_file(url, filename="mtbench.jsonl")
        questions_data = list(read_jsonl("mtbench.jsonl"))
        questions_data = questions_data

        questions = [
            {"question_1": q["turns"][0], "question_2": q["turns"][1]}
            for q in questions_data
        ]
        # MT-Bench doesn't have labels for accuracy computation
        labels = [None] * len(questions)

        if self.num_samples is not None:
            questions = questions[: self.num_samples]
            labels = labels[: self.num_samples]
        return questions, labels

    def create_sgl_function(self):
        """Create SGL function for MT-Bench (2-turn conversation)."""
        return create_multi_turn_sgl_function(
            function_name="answer_mt_bench",
            system_prompt=SYSTEM_PROMPT,
            num_turns=2,
            max_tokens=self.get_max_new_tokens(),
        )

    def get_answer_keys(self) -> List[str]:
        """Return answer keys for multi-turn conversation."""
        return ["answer_1", "answer_2"]
