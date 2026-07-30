# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""VLM (image+text) prompt preparation for server-side multimodal capture.

The multimodal online path needs, per sample:

- ``input_ids`` / ``loss_mask``: the FULLY image-expanded token sequence.
  These are what the trainer consumes and what the passthrough capture stores,
  so they must match the capture server's own expansion one-for-one.
- ``request_input_ids``: the same sequence with each image region collapsed
  back to a single ``<|image_pad|>`` placeholder. This is what the capture
  request sends; the patched server re-expands it in id space
  (``SGLANG_MM_AVOID_RETOKENIZE=1``), which is guaranteed to reproduce the
  client expansion because both sides run the target model's own HF processor
  on the same image.
- ``image_data``: a base64 image string for the capture request (the image
  itself never enters the feature store; only text tokens and captured hidden
  states do).

Limitations (v1): at most one image per sample; the image is attached to the
first user turn (matching the Qwen-VL chat layout); text-only samples are
supported in the same run (``image_data=None``). Image references are read
from ``image`` / ``image_path`` (string) or ``images`` (one-element list);
unreadable images and multi-image samples are fatal ``ImageDataError``s,
never a silent text-only downgrade.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple

from .prompt_builder import _iter_records
from .template import TEMPLATE_REGISTRY

_IMAGE_PLACEHOLDER = "<|image_pad|>"
_VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"


def _image_token_count(image_grid_thw, merge_size: int) -> int:
    """Merged token count of one image region from its processor grid."""

    grid = list(image_grid_thw[0])
    count = 1
    for dim in grid:
        count *= int(dim)
    return count // (merge_size * merge_size)


class ImageDataError(ValueError):
    """Fatal image-contract violation (unreadable image, multi-image sample).

    Raised instead of silently degrading an image-bearing sample to text-only
    training; the prompt-preparation loop re-raises it to abort loudly.
    """


def _extract_image_field(record: Dict[str, Any], *, source: str) -> Optional[str]:
    """Resolve the single image reference of one record, or None.

    Accepts ``image`` / ``image_path`` (string) and ``images`` (a one-element
    list, the convention used by common VLM corpora). A record carrying more
    than one image is a fatal error in v1 (single-image contract), and so is
    an ``images`` list whose element is not a string.
    """

    image_field = record.get("image") or record.get("image_path")
    images = record.get("images")
    if image_field and images:
        raise ImageDataError(
            f"{source}: record has both 'image' and 'images' fields; use one"
        )
    if images is None:
        return image_field
    if not isinstance(images, list):
        raise ImageDataError(
            f"{source}: 'images' must be a list, got {type(images).__name__}"
        )
    if len(images) == 0:
        return image_field
    if len(images) > 1:
        raise ImageDataError(
            f"{source}: multi-image samples are not supported (got "
            f"{len(images)} images); the v1 contract is one image per sample"
        )
    first = images[0]
    if not isinstance(first, str):
        raise ImageDataError(
            f"{source}: 'images[0]' must be a path or base64 string, got "
            f"{type(first).__name__}"
        )
    return first


def _load_image(image_field: Any, *, source: str):
    """Return (pil_image, base64_str) from a path / base64 / data-URI field."""

    from PIL import Image

    if not isinstance(image_field, str) or not image_field:
        raise ImageDataError(
            f"{source}: image field must be a file path or base64 string, got "
            f"{type(image_field).__name__}"
        )
    if os.path.isfile(image_field):
        with open(image_field, "rb") as image_file:
            raw = image_file.read()
        return Image.open(io.BytesIO(raw)).convert("RGB"), base64.b64encode(raw).decode(
            "ascii"
        )
    encoded = image_field
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ImageDataError(
            f"{source}: image field is neither an existing file nor valid base64"
        ) from exc
    return Image.open(io.BytesIO(raw)).convert("RGB"), encoded


def _render_conversation_text(
    tokenizer,
    conversations: List[Dict[str, Any]],
    *,
    with_image: bool,
    source: str,
) -> str:
    """Render one ShareGPT conversation with the target tokenizer's template.

    The image (if any) becomes an OpenAI-style list content on the first user
    turn so the model's own chat template emits the vision markers exactly as
    at serving time.
    """

    messages: List[Dict[str, Any]] = []
    image_attached = False
    for message in conversations:
        role = message.get("role", message.get("from", ""))
        content = message.get("content")
        if content is None:
            content = message.get("value", "")
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        if isinstance(content, list):
            # OpenAI-style list content: keep text parts; image parts are
            # re-attached by this function (single-image v1).
            content = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(content, str):
            raise ValueError(f"{source}: unsupported message content type")
        if with_image and role == "user" and not image_attached:
            content = [
                {"type": "image"},
                {"type": "text", "text": content},
            ]
            image_attached = True
        messages.append({"role": role, "content": content})
    if with_image and not image_attached:
        raise ValueError(f"{source}: image sample has no user turn to attach to")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _expand_image_region(
    ids: List[int],
    mask: List[int],
    *,
    pad_token_id: int,
    count: int,
    source: str,
) -> Tuple[List[int], List[int]]:
    """Splice the single image placeholder into ``count`` copies in id space."""

    positions = [index for index, token in enumerate(ids) if token == pad_token_id]
    if len(positions) != 1:
        raise ValueError(
            f"{source}: expected exactly 1 image placeholder token, found "
            f"{len(positions)}"
        )
    index = positions[0]
    expanded_ids = ids[:index] + [pad_token_id] * count + ids[index + 1 :]
    expanded_mask = mask[:index] + [0] * count + mask[index + 1 :]
    return expanded_ids, expanded_mask


def build_vlm_prompt_payloads(
    path: str,
    tokenizer,
    processor,
    *,
    chat_template: Optional[str],
    max_length: int,
    min_loss_tokens: int = 1,
    max_prompts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build prompt payloads for multimodal server capture from a JSONL file.

    Records follow the ShareGPT shape (``conversations`` column); an optional
    top-level ``image`` field carries one image (file path or base64). Returns
    control-plane dicts ``{"payload": {...}}`` mirroring
    :func:`specforge.data.prompt_builder.prepare_prompt_tasks`.
    """

    template = TEMPLATE_REGISTRY.get(chat_template) if chat_template else None
    pad_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_PLACEHOLDER)
    if not isinstance(pad_token_id, int):
        raise ValueError(
            f"tokenizer has no {_IMAGE_PLACEHOLDER!r} special token; is this a "
            "Qwen-VL-family tokenizer?"
        )
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))

    payloads: List[Dict[str, Any]] = []
    skipped = 0
    for line_number, record in _iter_records(path):
        source = f"{path}:{line_number}"
        try:
            prepared = _prepare_one_record(
                record,
                source=source,
                tokenizer=tokenizer,
                processor=processor,
                template=template,
                pad_token_id=pad_token_id,
                merge_size=merge_size,
                max_length=max_length,
                min_loss_tokens=min_loss_tokens,
            )
        except ImageDataError:
            # Image-contract violations are fatal: never degrade an
            # image-bearing sample to text-only training silently.
            raise
        except ValueError as exc:
            print(f"WARNING: skipping {source}: {exc}")
            skipped += 1
            continue
        if prepared is None:
            skipped += 1
            continue
        payloads.append(prepared)
        if max_prompts not in (None, 0) and len(payloads) >= max_prompts:
            break
    print(
        f"VLM prompt preparation done: {len(payloads)} prepared, "
        f"{skipped} skipped ({path})"
    )
    return payloads


def _prepare_one_record(
    record: Dict[str, Any],
    *,
    source: str,
    tokenizer,
    processor,
    template,
    pad_token_id: int,
    merge_size: int,
    max_length: int,
    min_loss_tokens: int,
) -> Optional[Dict[str, Any]]:
    from .preprocessing import preprocess_conversations

    conversations = record.get("conversations")
    if not conversations:
        raise ValueError("record has no 'conversations' field")
    image_field = _extract_image_field(record, source=source)

    text = _render_conversation_text(
        tokenizer,
        conversations,
        with_image=bool(image_field),
        source=source,
    )
    # Reuse the text stack: preformatted rendering -> ids + loss mask.
    parsed = preprocess_conversations(
        tokenizer,
        [text],
        template,
        max_length=max_length,
        is_preformatted=True,
    )
    if not parsed["input_ids"]:
        return None
    collapsed = parsed["input_ids"][0][0].tolist()
    mask = parsed["loss_mask"][0][0].tolist()

    image_data = None
    if image_field:
        pil_image, image_data = _load_image(image_field, source=source)
        processor_output = processor.image_processor(
            images=[pil_image], return_tensors="pt"
        )
        count = _image_token_count(
            processor_output["image_grid_thw"].tolist(), merge_size
        )
        ids, mask = _expand_image_region(
            collapsed,
            mask,
            pad_token_id=pad_token_id,
            count=count,
            source=source,
        )
        if len(ids) > max_length:
            # Truncating into an image region would desynchronize the client's
            # expanded ids from the server's own expansion; drop the sample.
            raise ValueError(
                f"expanded sequence length {len(ids)} exceeds max_length "
                f"{max_length} (image region cannot be truncated)"
            )
    else:
        ids = collapsed

    ids = ids[:max_length]
    mask = mask[:max_length]
    if sum(mask) < min_loss_tokens:
        return None
    return {
        "payload": {
            "input_ids": ids,
            "loss_mask": mask,
            "request_input_ids": collapsed,
            "image_data": image_data,
        }
    }


__all__ = ["ImageDataError", "build_vlm_prompt_payloads"]
