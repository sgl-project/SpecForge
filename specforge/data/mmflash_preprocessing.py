"""Prepare Qwen3.5 single-image teacher-forcing pairs for the MMFlash recipe."""

from __future__ import annotations

import base64
import binascii
import io
import os


def load_record_image(record, *, source):
    """Read the single image path or embedded image used by the scorer."""
    from PIL import Image

    image = record.get("image") or record.get("image_path")
    images = record.get("images")
    if images is not None:
        if not isinstance(images, list) or len(images) > 1 or (images and image):
            raise ValueError(f"{source}: expected at most one image per record")
        if images:
            image = images[0]
            if not image:
                raise ValueError(f"{source}: empty image reference")
    if image is None:
        return None
    if not isinstance(image, str) or not image:
        raise ValueError(f"{source}: expected an image path or base64 string")
    # An embedded image can be longer than any supported filename.
    if not image.startswith("data:") and os.path.isfile(image):
        with Image.open(image) as picture:
            return picture.convert("RGB")
    encoded = image.partition(",")[2] if image.startswith("data:") else image
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"{source}: image is neither a file nor valid base64") from exc
    with Image.open(io.BytesIO(raw)) as picture:
        return picture.convert("RGB")


def chat_messages(turns, *, has_image):
    """Normalize ShareGPT turns and remove only the image from the blind copy."""
    visible, blind = [], []
    attached = False
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("MMFlash conversation turns must be objects")
        role = turn.get("role", turn.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        text = turn.get("content", turn.get("value"))
        if not isinstance(text, str) or role not in ("system", "user", "assistant"):
            raise ValueError(
                "MMFlash expects text ShareGPT user/assistant/system turns"
            )
        content = text
        if has_image and role == "user" and not attached:
            if text.count("<image>") > 1:
                raise ValueError("expected one image placeholder")
            # Attach the top-level image to the first user turn. The authors'
            # ShareGPT rows may also specify an explicit placeholder.
            prefix, _, suffix = text.partition("<image>")
            if "<image>" not in text:
                prefix, suffix = "", text
            content = []
            if prefix.strip():
                content.append({"type": "text", "text": prefix.strip()})
            content.append({"type": "image"})
            if suffix.strip():
                content.append({"type": "text", "text": suffix.strip()})
            text = text.replace("<image>", "").lstrip()
            attached = True
        elif "<image>" in text:
            raise ValueError(
                "image placeholder outside the single image-bearing user turn"
            )
        visible.append({"role": role, "content": content})
        blind.append({"role": role, "content": text})
    if has_image and not attached:
        raise ValueError("image row has no user turn")
    return visible, blind


def _render(processor, messages):
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def assistant_delimiters(processor):
    """Obtain the stored-reply header including Qwen's empty thinking block."""
    marker = "MMFLASHASSISTANTMARKER"
    user = [{"role": "user", "content": "x"}]
    prefix = _render(processor, user)
    rendered = _render(processor, [*user, {"role": "assistant", "content": marker}])
    before, found, tail = rendered.partition(marker)
    if not found or not tail.startswith("<|im_end|>"):
        raise ValueError(
            "expected a Qwen stored assistant reply terminated by <|im_end|>"
        )
    header = before[len(os.path.commonprefix([prefix, before])) :]
    tokenizer = processor.tokenizer
    header_ids = tokenizer(header, add_special_tokens=False)["input_ids"]
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not header_ids or end is None or end == tokenizer.unk_token_id:
        raise ValueError(
            "could not derive assistant delimiters from the target processor"
        )
    return header_ids, end


def assistant_loss_mask(ids, header, end, *, last_turn_only):
    """Mark the stored assistant answer through its end-of-turn token."""
    starts = [
        i + len(header)
        for i, token in enumerate(ids)
        if token == header[0] and ids[i : i + len(header)] == header
    ]
    mask = [0] * len(ids)
    for start in starts[-1:] if last_turn_only else starts:
        try:
            stop = ids.index(end, start) + 1
        except ValueError as exc:
            raise ValueError("unterminated assistant response") from exc
        mask[start:stop] = [1] * (stop - start)
    return mask


def encode_record(record, processor, *, delimiters, last_turn_only, source):
    turns = record.get("conversations")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{source}: missing conversations")
    image = load_record_image(record, source=source)
    visible, blind = chat_messages(turns, has_image=image is not None)
    roles = [message["role"] for message in visible]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError(f"{source}: expected one user and one assistant turn")
    answer = visible[-1]["content"].lower()
    if (
        turns[-1].get("reasoning_content")
        or "<think>" in answer
        or "</think>" in answer
    ):
        raise ValueError(f"{source}: regenerate the assistant answer with thinking off")
    arguments = {}
    if image is not None:
        arguments["images"] = [image]
    seen = processor(
        text=[_render(processor, visible)], return_tensors="pt", **arguments
    )
    ids = seen["input_ids"][0].tolist()
    mask = assistant_loss_mask(ids, *delimiters, last_turn_only=last_turn_only)
    if not any(mask):
        raise ValueError(
            f"{source}: assistant header did not match the rendered answer"
        )
    unseen = None
    if image is not None:
        unseen = processor(text=[_render(processor, blind)], return_tensors="pt")
    return seen, unseen, mask
