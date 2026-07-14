# coding=utf-8
"""Qwen2.5-VL prompt preparation for the canonical rollout path."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from specforge.inference.media import MediaInputs, PreparedTargetInput
from specforge.runtime.contracts import PromptTask


def prepare_qwen_vl_record(
    processor: Any,
    *,
    image: Any,
    conversations: Sequence[Mapping[str, Any]],
    chat_template: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize one image conversation with the retained VLM preprocessor."""
    from specforge.data.preprocessing import preprocess_vlm_conversations
    from specforge.data.template import TEMPLATE_REGISTRY

    template = TEMPLATE_REGISTRY.get(chat_template)
    result = preprocess_vlm_conversations(
        processor,
        {"image": [image], "conversations": [list(conversations)]},
        template,
        max_length,
    )
    if len(result["input_ids"]) != 1:
        raise ValueError("VLM preprocessing did not produce exactly one sample")
    return {name: values[0] for name, values in result.items()}


class QwenVLInputPreparer:
    """Re-materialize pixels inside rollout while PromptTask stays metadata-only."""

    def __init__(self, processor: Any, chat_template: str) -> None:
        self.processor = processor
        self.chat_template = chat_template

    def prepare(self, task: PromptTask, device: str) -> PreparedTargetInput:
        payload = task.payload
        media = payload.get("media")
        if not isinstance(media, Mapping):
            raise ValueError("VLM PromptTask payload is missing media metadata")
        prepared = prepare_qwen_vl_record(
            self.processor,
            image=media["image"],
            conversations=media["conversations"],
            chat_template=self.chat_template,
            max_length=task.max_length,
        )

        input_ids = prepared["input_ids"].reshape(-1)
        loss_mask = prepared["loss_mask"].reshape(-1)
        expected_ids = torch.as_tensor(payload["input_ids"], dtype=torch.long)
        expected_loss = torch.as_tensor(payload["loss_mask"], dtype=torch.long)
        if not torch.equal(input_ids.cpu(), expected_ids):
            raise ValueError("VLM prompt tokenization changed between ingest and rollout")
        if not torch.equal(loss_mask.cpu(), expected_loss):
            raise ValueError("VLM loss mask changed between ingest and rollout")

        grid = prepared["image_grid_thw"]
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        return PreparedTargetInput(
            input_ids=input_ids.to(device=device, dtype=torch.long),
            attention_mask=prepared["attention_mask"]
            .reshape(-1)
            .to(device=device, dtype=torch.long),
            loss_mask=loss_mask.to(device=device, dtype=torch.long),
            media=MediaInputs(
                pixel_values=prepared["pixel_values"].to(device),
                image_grid_thw=(grid.to(device=device, dtype=torch.long),),
            ),
        )


__all__ = ["QwenVLInputPreparer", "prepare_qwen_vl_record"]
