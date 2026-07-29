import json
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import cached_file

logger = logging.getLogger(__name__)


def _pre_tokenizer_types(pre_tokenizer):
    """Recursively collect the ``type`` of a (possibly nested) pre-tokenizer spec."""
    types = set()
    if not isinstance(pre_tokenizer, dict):
        return types
    if pre_tokenizer.get("type"):
        types.add(pre_tokenizer["type"])
    for sub in pre_tokenizer.get("pretokenizers") or []:
        types |= _pre_tokenizer_types(sub)
    return types


def load_tokenizer(pretrained_model_name_or_path, **kwargs):
    """Load a tokenizer, working around a transformers v5 fast-tokenizer regression.

    Some repos (e.g. ``deepseek-ai/DeepSeek-V3.2``) declare a SentencePiece-style
    ``tokenizer_class`` (``LlamaTokenizerFast``) but actually ship a ByteLevel-BPE
    ``tokenizer.json``. Under transformers v4 ``AutoTokenizer`` loaded the saved
    ``tokenizer.json`` verbatim. Under v5 the subclass ``__init__`` rebuilds the
    tokenizer and overrides the saved ByteLevel pre-tokenizer with a Metaspace one,
    silently dropping word-boundary spaces (``"Who are you?"`` -> ``"Whoareyou?"``)
    and corrupting both training data and our regression references.

    When the loaded fast tokenizer's pre-tokenizer no longer matches the one
    serialized in ``tokenizer.json``, we reload faithfully via
    ``PreTrainedTokenizerFast``, which uses ``tokenizer.json`` as-is.
    """
    faithful_kwargs = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
    except (AttributeError, ValueError) as exc:
        # The old eager SGLang target import happened to register some remote
        # model configs (for example DeepSeek-V3.2) before tokenizer loading.
        # The refactored target-engine boundary intentionally removed that side
        # effect. A generic fast tokenizer can load tokenizer.json directly and
        # does not need the model config to be registered.
        logger.warning(
            "AutoTokenizer could not resolve %s (%s: %s); loading "
            "tokenizer.json with PreTrainedTokenizerFast instead.",
            pretrained_model_name_or_path,
            type(exc).__name__,
            exc,
        )
        return PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path, **faithful_kwargs
        )

    if not getattr(tokenizer, "is_fast", False):
        return tokenizer

    # Locate the serialized fast tokenizer (handles both local paths and the hub cache).
    passthrough = {
        k: kwargs[k]
        for k in ("revision", "token", "cache_dir", "local_files_only")
        if k in kwargs
    }
    try:
        tokenizer_json_path = cached_file(
            pretrained_model_name_or_path,
            "tokenizer.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **passthrough,
        )
    except Exception:
        tokenizer_json_path = None

    if not tokenizer_json_path or not os.path.isfile(tokenizer_json_path):
        return tokenizer

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        saved_pre_tokenizer = json.load(f).get("pre_tokenizer")
    loaded_pre_tokenizer = json.loads(tokenizer.backend_tokenizer.to_str()).get(
        "pre_tokenizer"
    )

    saved_types = _pre_tokenizer_types(saved_pre_tokenizer)
    loaded_types = _pre_tokenizer_types(loaded_pre_tokenizer)

    # The regression we guard against drops the saved ByteLevel pre-tokenizer
    # (which maps spaces to a marker) in favor of a SentencePiece Metaspace one,
    # corrupting word-boundary spacing. Only reload in that specific case so we
    # don't needlessly swap the tokenizer class for the (common) tokenizers whose
    # subclass tweaks the pre-tokenizer while keeping ByteLevel intact.
    if "ByteLevel" not in saved_types or "ByteLevel" in loaded_types:
        return tokenizer

    logger.warning(
        "Tokenizer class %s dropped the ByteLevel pre-tokenizer saved in "
        "tokenizer.json (loaded %s); reloading with PreTrainedTokenizerFast to "
        "preserve the saved tokenization.",
        type(tokenizer).__name__,
        sorted(loaded_types) or "none",
    )
    # PreTrainedTokenizerFast loads tokenizer.json verbatim and needs no remote code.
    return PreTrainedTokenizerFast.from_pretrained(
        pretrained_model_name_or_path, **faithful_kwargs
    )


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def get_device_type() -> str:
    """Auto-detect the available accelerator type.

    Priority:
    1. SPECFORGE_DEVICE environment variable
    2. NVIDIA CUDA (torch.cuda)
    3. Ascend NPU (torch.npu)
    4. CPU fallback
    """
    dt = os.environ.get("SPECFORGE_DEVICE", None)
    if dt:
        return dt
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


def get_local_device() -> torch.device:
    """Return the local torch.device for the current process rank."""
    device_type = get_device_type()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        return torch.device("cuda", local_rank)
    if device_type == "npu":
        return torch.device("npu", local_rank)
    return torch.device("cpu")


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_args_with_dots(args):
    if dist.get_rank() == 0:
        args_dict = vars(args)
        max_key_length = max(len(key) for key in args_dict.keys())
        total_width = 50

        print("\n -----------【args】-----------")
        for key, value in args_dict.items():
            key_str = f"{key:<{max_key_length}}"
            value_str = str(value)
            dot_count = total_width - len(key_str) - len(value_str)
            dot_fill = "·" * dot_count
            print(f"{key_str} {dot_fill} {value_str}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


def safe_conversations_generator(file_path):
    """
    Generator that:
    1. Extracts the 'conversations' field.
    2. Preserves all original fields within each message.
    3. [Key step] Converts all list/dict-type field values to strings to resolve mixed-type conflicts (e.g., for Arrow compatibility).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                raw_convs = row.get("conversations", [])

                # 1. Ensure 'conversations' is a list
                if not isinstance(raw_convs, list):
                    # If it's None or some unexpected type, treat as empty or skip
                    if raw_convs is None:
                        raw_convs = []
                    else:
                        # Edge case: 'conversations' is a plain string or non-iterable—skip this line
                        logger.warning(
                            f"Line {i + 1}: 'conversations' is not a list. Please check!"
                        )
                        continue

                cleaned_convs = []
                for msg in raw_convs:
                    # 2. Ensure each item in the list is a dictionary
                    if not isinstance(msg, dict):
                        # Skip if an element is not a dict (e.g., malformed like ["user", "hi"])
                        continue

                    # 3. [Core logic] Iterate over all fields in the message (role, content, tools, etc.)
                    new_msg = {}
                    for k, v in msg.items():
                        # If the value is a list or dict, serialize it to a JSON string
                        # This ensures Arrow treats the column as string type instead of list/struct
                        if isinstance(v, (list, dict)):
                            new_msg[k] = json.dumps(v, ensure_ascii=False)
                        else:
                            # Keep primitive types (str, int, float, bool, None) unchanged
                            new_msg[k] = v

                    cleaned_convs.append(new_msg)

                # Build result with conversations
                result = {"conversations": cleaned_convs}

                # Preserve 'tools' field if present
                if "tools" in row:
                    tools = row["tools"]
                    if tools is not None:
                        # If tools is a JSON string, parse it first
                        if isinstance(tools, str):
                            try:
                                tools = json.loads(tools)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Line {i + 1}: 'tools' is a string but not valid JSON, keeping as-is"
                                )
                                result["tools"] = tools
                                yield result
                                continue

                        # Serialize tools to JSON string for Arrow compatibility
                        # (same treatment as list/dict fields in conversations)
                        if isinstance(tools, (list, dict)):
                            result["tools"] = json.dumps(tools, ensure_ascii=False)
                        else:
                            # Primitive type, keep as-is
                            result["tools"] = tools
                    else:
                        result["tools"] = []

                yield result

            except Exception as e:
                logger.warning(f"Skipping line {i + 1}: {e}")
                continue
