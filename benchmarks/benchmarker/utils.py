"""
Utility functions for benchmark scripts.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sglang as sgl


@dataclass
class BenchmarkMetrics:
    """Container for benchmark performance metrics."""

    latency: float
    output_throughput: float
    accept_length: float
    accuracy: Optional[float] = None
    num_questions: int = 0
    num_valid_predictions: int = 0
    categorical_performance: Optional[Dict[str, "BenchmarkMetrics"]] = None


def compute_metrics(
    states: List[Any],
    latency: float,
    answer_key: str = "answer",
    additional_answer_keys: Optional[List[str]] = None,
) -> BenchmarkMetrics:
    """
    Compute performance metrics from SGLang states.

    Args:
        states: List of SGLang state objects from run_batch
        latency: Total latency in seconds
        answer_key: Primary key for answer in state meta info
        additional_answer_keys: Additional keys to include in token count (e.g., ["answer_1", "answer_2"])

    Returns:
        BenchmarkMetrics object with computed metrics
    """
    # Compute output tokens
    num_output_tokens = 0
    if additional_answer_keys:
        for key in [answer_key] + additional_answer_keys:
            num_output_tokens += sum(
                s.get_meta_info(key)["completion_tokens"] for s in states
            )
    else:
        num_output_tokens = sum(
            s.get_meta_info(answer_key)["completion_tokens"] for s in states
        )

    output_throughput = num_output_tokens / latency if latency > 0 else 0.0

    # Compute accept length (speculative decoding metric)
    has_verify = "spec_verify_ct" in states[0].get_meta_info(answer_key)
    if has_verify:
        num_verify_tokens = 0
        if additional_answer_keys:
            for key in [answer_key] + additional_answer_keys:
                num_verify_tokens += sum(
                    s.get_meta_info(key).get("spec_verify_ct", 0) for s in states
                )
        else:
            num_verify_tokens = sum(
                s.get_meta_info(answer_key).get("spec_verify_ct", 0) for s in states
            )

        if num_verify_tokens == 0:
            accept_length = 1.0
        else:
            accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    return BenchmarkMetrics(
        latency=latency,
        output_throughput=output_throughput,
        accept_length=accept_length,
        num_questions=len(states),
    )


def print_results(
    metrics_list: List[BenchmarkMetrics],
    benchmark_name: str,
    show_accuracy: bool = False,
):
    """
    Print benchmark results in a formatted way.

    Args:
        metrics_list: List of BenchmarkMetrics from multiple runs
        benchmark_name: Name of the benchmark
        show_accuracy: Whether to show accuracy metrics
    """
    avg_latency = np.mean([m.latency for m in metrics_list])
    avg_throughput = np.mean([m.output_throughput for m in metrics_list])
    avg_accept_length = np.mean([m.accept_length for m in metrics_list])

    print(f"\n{'='*50}")
    print(f"{benchmark_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Number of questions: {metrics_list[0].num_questions}")
    if show_accuracy:
        if metrics_list[0].accuracy is not None:
            avg_accuracy = np.mean(
                [m.accuracy for m in metrics_list if m.accuracy is not None]
            )
            print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        else:
            print(f"Average Accuracy: None")
    print(f"Average Latency: {avg_latency:.3f} s")
    print(f"Average Output throughput: {avg_throughput:.3f} token/s")
    print(f"Average Accept length: {avg_accept_length:.3f}")
    print(f"{'='*50}\n")


def create_simple_sgl_function(
    function_name: str = "get_answer",
    answer_key: str = "answer",
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None,
    user_prefix: Optional[str] = None,
) -> Callable:
    """
    Create a simple SGL function for single-turn Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        stop: Optional stop sequences
        user_prefix: Optional suffix to append to user message (appended after question)

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, question):
        if system_prompt:
            s += sgl.system(system_prompt)
        user_content = question
        if user_prefix:
            user_content = question + user_prefix
        s += sgl.user(user_content)
        gen_kwargs = {"max_tokens": max_tokens}
        if stop:
            gen_kwargs["stop"] = stop
        s += sgl.assistant(sgl.gen(answer_key, **gen_kwargs))

    sgl_func.__name__ = function_name
    return sgl_func


def create_few_shot_sgl_function(
    few_shot_examples: str,
    function_name: str = "few_shot_answer",
    answer_key: str = "answer",
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
) -> Callable:
    """
    Create an SGL function for few-shot learning.

    Args:
        few_shot_examples: String containing few-shot examples
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate
        stop: Optional stop sequences

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, question):
        s += few_shot_examples + question
        gen_kwargs = {"max_tokens": max_tokens}
        if stop:
            gen_kwargs["stop"] = stop
        s += sgl.gen(answer_key, **gen_kwargs)

    sgl_func.__name__ = function_name
    return sgl_func


def create_multi_turn_sgl_function(
    function_name: str = "multi_turn_answer",
    system_prompt: Optional[str] = None,
    num_turns: int = 2,
    max_tokens: int = 2048,
) -> Callable:
    """
    Create an SGL function for multi-turn conversations (e.g., MT-Bench with 2 turns).

    Args:
        function_name: Name of the function
        system_prompt: Optional system prompt
        num_turns: Number of conversation turns (default: 2)
        max_tokens: Maximum tokens to generate per turn

    Returns:
        SGL function decorated with @sgl.function
    """
    if num_turns == 2:
        # Most common case: 2-turn conversation
        @sgl.function
        def sgl_func(s, question_1, question_2):
            if system_prompt:
                s += sgl.system(system_prompt)
            s += sgl.user(question_1)
            s += sgl.assistant(sgl.gen("answer_1", max_tokens=max_tokens))
            s += sgl.user(question_2)
            s += sgl.assistant(sgl.gen("answer_2", max_tokens=max_tokens))

    else:
        # Generic case: create function with dynamic number of turns
        # Note: This requires the caller to pass arguments as a dict
        @sgl.function
        def sgl_func(s, **kwargs):
            if system_prompt:
                s += sgl.system(system_prompt)
            for i in range(num_turns):
                question_key = f"question_{i+1}"
                answer_key = f"answer_{i+1}"
                if question_key in kwargs:
                    s += sgl.user(kwargs[question_key])
                    s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))

    sgl_func.__name__ = function_name
    return sgl_func


def create_image_sgl_function(
    function_name: str = "get_image_answer",
    answer_key: str = "answer",
    max_tokens: int = 2048,
) -> Callable:
    """
    Create an SGL function for image-based Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, image_path, question, **kwargs):
        """
        The body of the SGL function: constructs a multimodal conversation flow.

        - First, it inputs an image + text question as 'user'.
        - Then, it generates an answer as 'assistant', binding the response to the specified `answer_key`.

        Note: sgl.image() automatically encodes the image into a format supported by the model for multimodal input.
        """
        # User input: Image + Text question
        s += sgl.user(sgl.image(image_path) + question)
        s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))

    sgl_func.__name__ = function_name
    return sgl_func
