"""
Base class for benchmark implementations.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang import set_default_backend
from sglang.global_config import global_config
from sglang.test.test_utils import select_sglang_backend

from .utils import compute_metrics, set_chat_template_context

# SGLang's lang frontend traces every @sgl.function with placeholder Argument
# objects to extract a common prefix for caching. That tracing is incompatible
# with apply_chat_template (it sees content=Argument(...) and raises). We
# disable it here so each batch element is rendered with real values.
global_config.enable_precache_with_tracing = False


class Benchmarker(ABC):
    """
    Base class for benchmark implementations.

    Subclasses should implement:
    - load_data(): Load and preprocess dataset
    - create_sgl_function(): Create the SGL function for inference

    Optional overrides:
    - extract_answer(): Extract answer from model output (if needed)
    - compute_accuracy(): Compute accuracy metric (if applicable)
    - get_answer_keys(): Get list of answer keys for multi-turn conversations

    Args:
        num_samples: The number of samples to run the benchmark on. If not provided, all questions will be used.
        subset: The subset of the dataset to run the benchmark on. If not provided, all subsets will be used.
    """

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        self.num_samples = num_samples
        self.subset = subset

    @abstractmethod
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Load and preprocess the dataset.

        Returns:
            Tuple of (questions, labels) where:
            - questions: List of question dicts for SGL function
            - labels: List of ground truth labels (can be None if not applicable)
        """
        raise NotImplementedError

    @abstractmethod
    def create_sgl_function(self) -> Callable:
        """
        Create the SGL function for inference.

        Returns:
            SGL function decorated with @sgl.function
        """
        raise NotImplementedError

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[Any]:
        """
        Extract answer from model output.

        Args:
            output: Raw model output string
            label: Optional ground truth label for reference

        Returns:
            Extracted answer, or None if extraction fails
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Compute accuracy metric.

        Args:
            predictions: List of predicted answers
            labels: List of ground truth labels

        Returns:
            Accuracy score (0-1), or None if not applicable
        """
        return None

    def get_answer_keys(self) -> Optional[List[str]]:
        """
        Get list of answer keys for multi-turn conversations.

        Returns:
            List of answer keys (e.g., ["answer_1", "answer_2"]), or None for single-turn
        """
        return None

    def get_max_new_tokens(self) -> int:
        """
        Get maximum number of new tokens to generate.

        Returns:
            Maximum tokens (default: 2048)
        """
        return 2048

    def run(
        self,
        host: str,
        port: int,
        batch_size: int,
        max_new_tokens: int = None,
        num_runs: int = 1,
        generations_path: Optional[str] = None,
        record_extras: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """
        Run the benchmark evaluation.

        This method handles the common workflow:
        1. Initialize backend
        2. Load data
        3. Create SGL function
        4. Run inference loops
        5. Compute metrics
        6. Print results

        Args:
            host (str): The host of the SGLang server
            port (int): The port of the SGLang server
            batch_size (int): The number of prompts to process in parallel
            num_samples (int): The number of samples to run the benchmark on. If not provided, all samples will be used.
            max_new_tokens (int): Maximum number of new tokens to generate, default is 2048
            num_runs (int): The number of times to run this benchmark, default is 1. You can set it to a larger number if you want to get more stable results.
            generations_path (str): If provided, append per-prompt generations
                as JSONL to this path (one record per question per run). The
                caller owns the file lifecycle (truncate/rotate); this method
                only appends.
            record_extras (dict): Extra fields merged into every generation
                record before it is written (e.g. ``{"benchmark": "mtbench"}``).
            model_path (str): If set, prompts are pre-rendered with this
                model's HF chat_template (bypasses SGLang's stale frontend
                registry that mis-handles gpt-oss harmony, Qwen3, etc.).
            temperature (float): Sampling temperature passed to run_batch.
                0.0 keeps the verifier on the greedy kernel; >0 uses the
                rejection-sampling spec verify path.
            reasoning_effort (str): Forwarded to apply_chat_template (gpt-oss
                harmony picks it up as `Reasoning: <level>`). None = template
                default (medium for gpt-oss).
            enable_thinking (bool): Forwarded to apply_chat_template (Qwen3
                respects this). None = template default (True for Qwen3).
        """
        set_chat_template_context(
            model_path,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        # Initialize backend
        sglang_args = Namespace(host=host, port=port, backend="srt-no-parallel")
        set_default_backend(select_sglang_backend(sglang_args))

        # Load data
        questions, labels = self.load_data()
        if len(questions) == 0:
            print("No valid questions found. Please check the dataset format.")
            return

        # Create SGL function
        sgl_function = self.create_sgl_function()

        # Run evaluation loops
        metrics_list = []
        answer_keys = self.get_answer_keys()
        max_new_tokens = max_new_tokens or self.get_max_new_tokens()

        if generations_path is not None:
            os.makedirs(os.path.dirname(generations_path) or ".", exist_ok=True)

        for run_idx in range(num_runs):
            tic = time.perf_counter()
            states = sgl_function.run_batch(
                questions,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_threads=batch_size,
                progress_bar=True,
            )
            latency = time.perf_counter() - tic

            # Extract predictions
            predictions = []
            primary_answer_key = answer_keys[0] if answer_keys else "answer"
            keys_to_dump = answer_keys or [primary_answer_key]
            cached_outputs: List[Dict[str, Any]] = []
            for i in range(len(states)):
                outputs: Dict[str, Any] = {}
                for key in keys_to_dump:
                    try:
                        outputs[key] = states[i][key]
                    except KeyError:
                        outputs[key] = None
                cached_outputs.append(outputs)

                output = outputs.get(primary_answer_key)
                if isinstance(output, str):
                    extracted = self.extract_answer(
                        output,
                        (labels[i] if labels and i < len(labels) else None),
                    )
                else:
                    extracted = output
                predictions.append(extracted)

            if generations_path is not None:
                with open(generations_path, "a") as gf:
                    for i in range(len(states)):
                        record: Dict[str, Any] = {}
                        if record_extras:
                            record.update(record_extras)
                        record.update(
                            {
                                "run": run_idx,
                                "index": i,
                                "question": questions[i],
                                "outputs": cached_outputs[i],
                                "prediction": predictions[i],
                                "label": (
                                    labels[i] if labels and i < len(labels) else None
                                ),
                            }
                        )
                        gf.write(json.dumps(record, default=str) + "\n")

            # Compute accuracy if applicable
            accuracy = None
            # Check if we have a labels list (even if all labels are None)
            has_labels_list = labels and len(labels) > 0

            if has_labels_list:
                # Always call compute_accuracy if we have a labels list
                # This allows it to return None, which will be displayed in print_results
                accuracy = self.compute_accuracy(predictions, labels)
                if accuracy is not None:
                    valid_count = sum(1 for p in predictions if p is not None)
                    if valid_count < len(predictions):
                        print(
                            f"Warning: {len(predictions) - valid_count} predictions could not be extracted."
                        )

            # Compute performance metrics
            metrics = compute_metrics(
                states,
                latency,
                answer_key=primary_answer_key,
                additional_answer_keys=(
                    answer_keys[1:] if answer_keys and len(answer_keys) > 1 else None
                ),
            )
            # Always set accuracy if we have a labels list (even if compute_accuracy returns None)
            # This allows print_results to show None when compute_accuracy returns None
            if has_labels_list:
                metrics.accuracy = (
                    accuracy  # Can be None if compute_accuracy returns None
                )
                if accuracy is not None:
                    metrics.num_valid_predictions = sum(
                        1 for p in predictions if p is not None
                    )

            metrics_list.append(metrics)
        return metrics_list
