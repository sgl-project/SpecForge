# Copyright 2024 SpecForge Authors
# Licensed under the Apache License, Version 2.0

"""
Qwen2-VL Template for SpecForge.

This module provides the template class for Qwen2-VL models, implementing
model-specific encoding and collation logic.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from specforge.data.vlm_template.base import VLMTemplate


class QwenVLTemplate(VLMTemplate):
    """
    Template for Qwen2-VL models (Qwen2.5-VL-7B-Instruct, etc.).
    
    Qwen2-VL specific features:
    - Uses qwen_vl_utils for image processing
    - Outputs pixel_values and image_grid_thw
    - Uses standard image placeholder format
    """
    
    def __init__(
        self,
        processor_path: str,
        max_length: int = 2048,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        trust_remote_code: bool = True,
        assistant_header: str = "<|im_start|>assistant\n",
        end_of_turn_token: str = "<|im_end|>\n",
    ):
        super().__init__(
            processor_path=processor_path,
            max_length=max_length,
            system_prompt=system_prompt,
            trust_remote_code=trust_remote_code,
        )
        self.assistant_header = assistant_header
        self.end_of_turn_token = end_of_turn_token
    
    def encode(self, example: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode a single Qwen2-VL example.
        
        Args:
            example: Dictionary containing:
                - image: Single path or list of image paths
                - conversations: List of {"role": str, "content": str}
        
        Returns:
            Dictionary containing:
                - input_ids: torch.Tensor of shape (1, seq_len)
                - attention_mask: torch.Tensor of shape (1, seq_len)
                - loss_mask: torch.Tensor of shape (1, seq_len)
                - pixel_values: torch.Tensor of image features
                - image_grid_thw: torch.Tensor of grid info
        """
        # Ensure qwen_vl_utils is available
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "qwen_vl_utils is required for Qwen VL preprocessing. "
                "Please install it: pip install qwen-vl-utils"
            )
        
        # Get image and conversations
        image_input = example.get("image")
        conversations = example.get("conversations", [])
        
        if not conversations:
            return None
        
        # Skip first turn if not from user
        if conversations[0]["role"] != "user":
            conversations = conversations[1:]
        
        if not conversations:
            return None
        
        # Build messages in Qwen format
        messages = self._build_messages(conversations, image_input)
        
        # Apply chat template
        conversation = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Get vision info using qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(messages)
        
        if image_inputs is None:
            logging.warning("Skipping sample: no image inputs")
            return None
        
        # Process with Qwen processor
        try:
            encoding = self.processor(
                text=[conversation],
                images=image_inputs,
                videos=video_inputs,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        except Exception as e:
            logging.warning(f"Skipping sample due to processor error: {e}")
            return None
        
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        pixel_values = encoding.pixel_values
        image_grid_thw = encoding.image_grid_thw[0]
        
        # Get decoded text for loss mask computation
        decoded_conversation = self.processor.tokenizer.decode(
            input_ids, skip_special_tokens=False
        )
        
        # Compute loss mask
        loss_mask = self._compute_loss_mask(decoded_conversation, offsets)
        
        if loss_mask.sum() == 0:
            logging.warning("Skipping sample: no assistant response")
            return None
        
        return {
            "input_ids": input_ids[None, :],
            "attention_mask": torch.ones_like(input_ids)[None, :],
            "loss_mask": loss_mask[None, :],
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw[None, :],
        }
    
    def _build_messages(
        self, 
        conversations: List[Dict[str, str]], 
        image_input: Any
    ) -> List[Dict[str, Any]]:
        """Build messages in Qwen format with image content."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for i, conv in enumerate(conversations):
            role = conv["role"]
            content = conv["content"]
            
            if role == "user" and i == 0 and image_input:
                # First user message with image
                messages.append({
                    "role": role,
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": content},
                    ],
                })
            else:
                messages.append({"role": role, "content": content})
        
        return messages
    
    def _compute_loss_mask(
        self, 
        text: str, 
        offsets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss mask for assistant responses using regex matching.
        """
        loss_mask = torch.zeros(len(offsets), dtype=torch.long)
        
        # Build pattern for assistant responses
        assistant_pattern = (
            re.escape(self.end_of_turn_token + self.assistant_header.split('\n')[0])
            + r".*?"
            + r"(.*?)(?="
            + re.escape(self.end_of_turn_token)
            + r"|$)"
        )
        
        # Alternative simpler pattern
        pattern = re.escape(self.assistant_header) + r"(.*?)(?=" + re.escape(self.end_of_turn_token) + r"|$)"
        
        for match in re.finditer(pattern, text, re.DOTALL):
            start_char = match.start(1)
            end_char = match.end(1)
            
            # Map character positions to token positions
            for token_idx, (token_start, token_end) in enumerate(offsets.tolist()):
                if token_start is None or token_end is None:
                    continue
                if token_start >= start_char and token_end <= end_char:
                    loss_mask[token_idx] = 1
                elif token_start < end_char and token_end > start_char:
                    loss_mask[token_idx] = 1
        
        return loss_mask
    
    def collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of Qwen2-VL features.
        
        Args:
            features: List of encoded examples
        
        Returns:
            Batched dictionary with:
                - input_ids: (B, max_len)
                - attention_mask: (B, max_len)
                - loss_mask: (B, max_len)
                - pixel_values: Concatenated pixel values
                - image_grid_thw: Concatenated grid info
                - hidden_state: None
                - target: None
        """
        # Filter out None values
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("All features are None after filtering")
        
        max_length = max(item["input_ids"].shape[1] for item in features)
        
        # Pad and stack text tensors
        batch_input_ids = torch.cat([
            self._pad_tensor_2d(item["input_ids"], max_length) 
            for item in features
        ])
        batch_attention_mask = torch.cat([
            self._pad_tensor_2d(item["attention_mask"], max_length) 
            for item in features
        ])
        batch_loss_mask = torch.cat([
            self._pad_tensor_2d(item["loss_mask"], max_length) 
            for item in features
        ])
        
        # Concatenate image tensors
        batch_pixel_values = torch.cat(
            [item["pixel_values"] for item in features], dim=0
        )
        batch_image_grid_thw = torch.cat(
            [item["image_grid_thw"] for item in features], dim=0
        )
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
            "hidden_state": None,
            "target": None,
        }
