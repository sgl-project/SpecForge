# Copyright 2024 SpecForge Authors
# Licensed under the Apache License, Version 2.0

"""
MiniCPM-V Template for SpecForge.

This module provides the template class for MiniCPM-V models, implementing
model-specific encoding and collation logic.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from specforge.data.vlm_template.base import VLMTemplate


class MiniCPMVTemplate(VLMTemplate):
    """
    Template for MiniCPM-V models (MiniCPM-V-4, etc.).
    
    MiniCPM-V specific features:
    - Uses slice-based image encoding
    - Outputs pixel_values as nested list of tensors
    - Uses tgt_sizes and image_bound for image positioning
    - Uses (<image>./</image>) as image placeholder
    """
    
    # MiniCPM-V specific token IDs
    IM_START_ID = 73441  # <|im_start|>
    IM_END_ID = 73440    # <|im_end|>
    ASSISTANT_ID = 16434 # assistant
    NEWLINE_ID = 5       # \n
    UNK_TOKEN_ID = 0     # <unk>
    
    # Placeholder formats
    STANDARD_PLACEHOLDER = "<image>"
    MINICPM_PLACEHOLDER = "(<image>./</image>)"
    
    def __init__(
        self,
        processor_path: str,
        max_length: int = 2048,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        trust_remote_code: bool = True,
    ):
        super().__init__(
            processor_path=processor_path,
            max_length=max_length,
            system_prompt=system_prompt,
            trust_remote_code=trust_remote_code,
        )
    
    def encode(self, example: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Encode a single MiniCPM-V example.
        
        Args:
            example: Dictionary containing:
                - image: Single path or list of image paths
                - conversations: List of {"role": str, "content": str}
        
        Returns:
            Dictionary containing:
                - input_ids: torch.Tensor of shape (1, seq_len)
                - attention_mask: torch.Tensor of shape (1, seq_len)
                - loss_mask: torch.Tensor of shape (1, seq_len)
                - pixel_values: List of tensors (one per image slice)
                - tgt_sizes: torch.Tensor of target sizes
                - image_bound: torch.Tensor of image boundaries
                - position_ids: torch.Tensor of position IDs
        """
        # Get image and conversations
        image_input = example.get("image")
        conversations = example.get("conversations", [])
        
        if not conversations:
            return None
        
        # Load images
        images = self._load_images(image_input)
        num_images = len(images)
        
        # Skip first turn if not from user
        if conversations[0]["role"] != "user":
            conversations = conversations[1:]
        
        if not conversations:
            return None
        
        # Build messages with proper placeholders
        messages = self._build_messages(conversations, num_images)
        
        # Process with MiniCPM-V processor
        try:
            encoding = self._process_with_processor(messages, images)
        except Exception as e:
            logging.warning(f"Skipping sample due to processor error: {e}")
            return None
        
        input_ids = encoding.input_ids[0]
        
        # Compute loss mask for assistant responses
        loss_mask = self._compute_loss_mask(input_ids)
        
        # Check for truncation or empty response
        if loss_mask is None or loss_mask.sum() == 0:
            logging.warning("Skipping sample: truncated or no assistant response")
            return None
        
        # Extract MiniCPM-V specific fields
        pixel_values = encoding.pixel_values[0] if hasattr(encoding, "pixel_values") and encoding.pixel_values else []
        tgt_sizes = encoding.tgt_sizes[0] if hasattr(encoding, "tgt_sizes") and encoding.tgt_sizes else None
        image_bound = encoding.image_bound[0] if hasattr(encoding, "image_bound") and encoding.image_bound else None
        position_ids = encoding.position_ids if hasattr(encoding, "position_ids") else None
        
        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(len(input_ids), dtype=torch.long)[None, :]
        
        return {
            "input_ids": input_ids[None, :],
            "attention_mask": torch.ones_like(input_ids)[None, :],
            "loss_mask": loss_mask[None, :],
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
            "position_ids": position_ids,
        }
    
    def _build_messages(
        self, 
        conversations: List[Dict[str, str]], 
        num_images: int
    ) -> List[Dict[str, str]]:
        """Build messages with proper image placeholders."""
        messages = []
        
        # Check if content already has placeholders
        first_content = conversations[0]["content"] if conversations else ""
        has_placeholder = (
            self.STANDARD_PLACEHOLDER in first_content or 
            self.MINICPM_PLACEHOLDER in first_content
        )
        
        for i, conv in enumerate(conversations):
            role = conv["role"]
            content = conv["content"]
            
            # Convert standard placeholder to MiniCPM-V format
            if self.STANDARD_PLACEHOLDER in content:
                content = content.replace(
                    self.STANDARD_PLACEHOLDER, 
                    self.MINICPM_PLACEHOLDER
                )
            
            # Add image placeholders to first user message if not present
            if role == "user" and i == 0 and num_images > 0 and not has_placeholder:
                placeholders = "\n".join([self.MINICPM_PLACEHOLDER] * num_images)
                content = f"{placeholders}\n{content}"
            
            messages.append({"role": role, "content": content})
        
        # Add system prompt
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        
        return messages
    
    def _process_with_processor(
        self, 
        messages: List[Dict[str, str]], 
        images: List
    ) -> Any:
        """Process messages and images with MiniCPM-V processor."""
        # Format conversation text
        conversation = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Process with MiniCPM-V processor
        # Note: MiniCPM-V processor has a bug where images=None causes UnboundLocalError
        encoding = self.processor(
            text=[conversation],
            images=[images] if images else [[]],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return encoding
    
    def _compute_loss_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute loss mask for assistant responses.
        
        Pattern: <|im_start|> assistant \n ... <|im_end|>
        """
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
        is_truncated = False
        idx = 0
        
        while idx < len(input_ids) - 2:
            # Check for <|im_start|> assistant \n sequence
            if (input_ids[idx] == self.IM_START_ID and 
                input_ids[idx + 1] == self.ASSISTANT_ID and 
                input_ids[idx + 2] == self.NEWLINE_ID):
                # Content starts at idx+3
                start = idx + 3
                # Find ending <|im_end|>
                end = start
                while end < len(input_ids) and input_ids[end] != self.IM_END_ID:
                    end += 1
                
                # Check if truncated
                if end >= len(input_ids):
                    is_truncated = True
                    break
                
                # Mark assistant response tokens (including <|im_end|>)
                loss_mask[start:end+1] = 1
                idx = end + 1
            else:
                idx += 1
        
        if is_truncated:
            return None
        
        return loss_mask
    
    def collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of MiniCPM-V features.
        
        Args:
            features: List of encoded examples
        
        Returns:
            Batched dictionary with:
                - input_ids: (B, max_len)
                - attention_mask: (B, max_len)
                - loss_mask: (B, max_len)
                - pixel_values: List of pixel value lists
                - tgt_sizes: List of tgt_sizes
                - image_bound: List of image_bounds
                - position_ids: (B, max_len)
                - hidden_state: None (filled by online training)
                - target: None (filled by online training)
        """
        # Filter out None values
        features = [f for f in features if f is not None]
        if not features:
            # Return empty batch - training loop should skip this
            import logging
            logging.warning("All features are None in this batch, returning empty batch")
            return None
        
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
        batch_position_ids = torch.cat([
            self._pad_tensor_2d(item["position_ids"], max_length) 
            for item in features
        ])
        
        # MiniCPM-V specific: keep as lists for variable sizes
        batch_pixel_values = [item["pixel_values"] for item in features]
        batch_tgt_sizes = [item["tgt_sizes"] for item in features]
        batch_image_bound = [item["image_bound"] for item in features]
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "pixel_values": batch_pixel_values,
            "tgt_sizes": batch_tgt_sizes,
            "image_bound": batch_image_bound,
            "position_ids": batch_position_ids,
            "hidden_state": None,
            "target": None,
        }
