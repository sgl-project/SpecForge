"""
Utility functions for benchmark scripts.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sglang as sgl

# SGLang's @sgl.function frontend formats sgl.user/system/assistant via its
# own hand-rolled registry (python/sglang/lang/chat_template.py), which
# doesn't cover gpt-oss harmony or Qwen3 (its matcher requires "qwen.*(chat|
# instruct)"). When a tokenizer is set here, the helpers pre-render prompts
# with the HF tokenizer's real chat_template and bypass the stale registry.
_tokenizer = None
_model_path: Optional[str] = None
# Forwarded to apply_chat_template; e.g. {"enable_thinking": False} for Qwen3
# or {"reasoning_effort": "low"} for gpt-oss.
_chat_template_kwargs: Dict[str, Any] = {}


def set_chat_template_context(
    model_path: Optional[str],
    *,
    reasoning_effort: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> None:
    """Load the tokenizer for `model_path` and remember chat-template kwargs."""
    global _tokenizer, _model_path, _chat_template_kwargs
    _chat_template_kwargs = {}
    if reasoning_effort is not None:
        _chat_template_kwargs["reasoning_effort"] = reasoning_effort
    if enable_thinking is not None:
        _chat_template_kwargs["enable_thinking"] = enable_thinking
    if model_path == _model_path:
        return
    if model_path is None:
        _tokenizer, _model_path = None, None
        return
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _model_path = model_path


def _has_chat_template() -> bool:
    return _tokenizer is not None and getattr(_tokenizer, "chat_template", None) is not None


def _render_chat(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
    rendered = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **_chat_template_kwargs,
    )
    # gpt-oss has no native enable_thinking knob; harmony's `analysis` channel
    # only fires if the model picks it after <|start|>assistant. Pre-fill the
    # final-channel marker to force the model straight into a final answer.
    if (
        add_generation_prompt
        and _chat_template_kwargs.get("enable_thinking") is False
        and rendered.endswith("<|start|>assistant")
    ):
        rendered += "<|channel|>final<|message|>"
    return rendered


# gpt-oss wraps assistant output in <|channel|>final<|message|>...<|end|>;
# the template re-wraps if we replay that verbatim, so strip back to the body.
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", re.DOTALL
)


def _assistant_for_replay(text: str) -> str:
    matches = _HARMONY_FINAL_RE.findall(text)
    if matches:
        return matches[-1]
    # No channel scaffolding in the output (e.g. when we pre-filled the
    # final-channel marker into the prompt). Strip trailing harmony enders.
    text = text.rstrip()
    for end in ("<|end|>", "<|return|>"):
        if text.endswith(end):
            text = text[: -len(end)].rstrip()
    return text


def _delta(cur: str, target: str) -> str:
    """Suffix of target to append after cur. Falls back to longest-common-prefix."""
    if target.startswith(cur):
        return target[len(cur):]
    n = 0
    cap = min(len(cur), len(target))
    while n < cap and cur[n] == target[n]:
        n += 1
    return target[n:]


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


def _single_turn(
    s, user_content: str, answer_key: str, gen_kwargs: dict, system_prompt: Optional[str]
):
    """Append a single user turn + assistant generation to the SGL state."""
    if _has_chat_template():
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_content})
        s += _render_chat(msgs, add_generation_prompt=True)
        s += sgl.gen(answer_key, **gen_kwargs)
    else:
        if system_prompt:
            s += sgl.system(system_prompt)
        s += sgl.user(user_content)
        s += sgl.assistant(sgl.gen(answer_key, **gen_kwargs))


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
    gen_kwargs = {"max_tokens": max_tokens, **({"stop": stop} if stop else {})}

    @sgl.function
    def sgl_func(s, question):
        _single_turn(s, question + (user_prefix or ""), answer_key, gen_kwargs, system_prompt)

    sgl_func.__name__ = function_name
    return sgl_func


def create_few_shot_sgl_function(
    few_shot_messages: Optional[List[Dict[str, str]]] = None,
    *,
    few_shot_examples: Optional[str] = None,
    function_name: str = "few_shot_answer",
    system_prompt: Optional[str] = None,
    answer_key: str = "answer",
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
) -> Callable:
    """SGL function for few-shot learning.

    Prefer `few_shot_messages` (alternating user/assistant turns) — those get
    prepended as real conversation history through the chat template.
    `few_shot_examples` is the legacy concatenated-string form.
    """
    if few_shot_messages is None and few_shot_examples is None:
        raise ValueError("provide either few_shot_messages or few_shot_examples")
    gen_kwargs = {"max_tokens": max_tokens, **({"stop": stop} if stop else {})}

    @sgl.function
    def sgl_func(s, question):
        if few_shot_messages is not None and _has_chat_template():
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.extend(few_shot_messages)
            msgs.append({"role": "user", "content": question})
            s += _render_chat(msgs, add_generation_prompt=True)
            s += sgl.gen(answer_key, **gen_kwargs)
        else:
            # No chat template (or caller passed legacy string) — smush back
            # into a single user-message blob.
            if few_shot_messages is not None:
                blob = ""
                for m in few_shot_messages:
                    if m["role"] == "user":
                        blob += "Question: " + m["content"] + "\n"
                    else:
                        blob += "Answer: " + m["content"] + "\n\n"
                user_content = blob + "Question: " + question + "\nAnswer:"
            else:
                user_content = few_shot_examples + question
            _single_turn(s, user_content, answer_key, gen_kwargs, system_prompt)

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

    @sgl.function
    def sgl_func(s, **kwargs):
        if _has_chat_template():
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            cum_text = ""
            for i in range(num_turns):
                qk, ak = f"question_{i+1}", f"answer_{i+1}"
                if qk not in kwargs:
                    continue
                msgs.append({"role": "user", "content": kwargs[qk]})
                rendered = _render_chat(msgs, add_generation_prompt=True)
                s += _delta(cum_text, rendered)
                s += sgl.gen(ak, max_tokens=max_tokens)
                msgs.append({"role": "assistant", "content": _assistant_for_replay(s[ak])})
                cum_text = rendered + s[ak]
        else:
            if system_prompt:
                s += sgl.system(system_prompt)
            for i in range(num_turns):
                qk, ak = f"question_{i+1}", f"answer_{i+1}"
                if qk in kwargs:
                    s += sgl.user(kwargs[qk])
                    s += sgl.assistant(sgl.gen(ak, max_tokens=max_tokens))

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
