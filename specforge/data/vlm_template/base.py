# Copyright 2024 SpecForge Authors
# Licensed under the Apache License, Version 2.0

"""
Base VLM Template class.

This module provides the base template class for VLM models. Each VLM model
should have its own template subclass that implements model-specific encoding
and collation logic.

Design inspired by ms-swift's Template architecture:
- dataset.map() only does message formatting (no image processing)
- Template.encode() handles image loading and tokenization
- Template.collate() handles batch padding and organization
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class VLMTemplate(ABC):
    """
    Base class for VLM templates.
    
    Each VLM model (MiniCPM-V, Qwen2-VL, etc.) should have its own template
    subclass that implements:
    - encode(): Process a single example (load images, tokenize, compute loss_mask)
    - collate(): Batch multiple encoded examples together
    
    This design enables:
    - Clean separation of model-specific logic
    - Natural multiprocessing support (processor loaded in main process)
    - Easy extension for new VLM models
    """
    
    def __init__(
        self,
        processor_path: str,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        """
        Initialize the VLM template.
        
        Args:
            processor_path: Path to load the processor from (usually same as model path)
            max_length: Maximum sequence length
            system_prompt: Optional system prompt to prepend
            trust_remote_code: Whether to trust remote code when loading processor
        """
        self.processor_path = processor_path
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._tokenizer = None
    
    @property
    def processor(self):
        """Lazy load processor on first access."""
        if self._processor is None:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.processor_path,
                trust_remote_code=self.trust_remote_code,
            )
        return self._processor
    
    @property
    def tokenizer(self):
        """Get tokenizer from processor."""
        if self._tokenizer is None:
            self._tokenizer = self.processor.tokenizer
        return self._tokenizer
    
    @abstractmethod
    def encode(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode a single example.
        
        This method should:
        1. Load images from paths
        2. Build messages with proper placeholders
        3. Call processor to get input_ids, pixel_values, etc.
        4. Compute loss_mask for assistant responses
        
        Args:
            example: Dictionary containing:
                - image: Single path or list of image paths
                - conversations: List of {"role": str, "content": str}
        
        Returns:
            Dictionary containing:
                - input_ids: torch.Tensor of shape (1, seq_len)
                - attention_mask: torch.Tensor of shape (1, seq_len)
                - loss_mask: torch.Tensor of shape (1, seq_len)
                - pixel_values: Model-specific image tensors
                - ... (other model-specific fields)
        """
        raise NotImplementedError
    
    @abstractmethod
    def collate(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of encoded features.
        
        This method should:
        1. Pad sequences to the same length
        2. Stack/concatenate tensors appropriately
        3. Handle model-specific fields (pixel_values, image_bound, etc.)
        
        Args:
            features: List of encoded examples from encode()
        
        Returns:
            Batched dictionary ready for model forward pass
        """
        raise NotImplementedError
    
    def _load_images(self, image_input: Any) -> List:
        """
        Load images from paths or return PIL Images directly.
        
        Args:
            image_input: Single path/PIL.Image or list of paths/PIL.Images
        
        Returns:
            List of PIL.Image objects
        """
        from PIL import Image
        
        if image_input is None:
            return []
        
        if not isinstance(image_input, list):
            image_input = [image_input]
        
        images = []
        for img in image_input:
            if isinstance(img, str):
                images.append(Image.open(img).convert("RGB"))
            elif hasattr(img, 'convert'):  # PIL.Image
                images.append(img.convert("RGB"))
            else:
                images.append(img)
        
        return images
    
    def _pad_tensor_2d(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad 2D tensor (B, L) to target length."""
        if tensor.shape[1] >= target_length:
            return tensor
        padding = torch.zeros(
            tensor.shape[0], 
            target_length - tensor.shape[1], 
            dtype=tensor.dtype
        )
        return torch.cat([tensor, padding], dim=1)
    
    def _pad_tensor_3d(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """Pad 3D tensor (B, L, D) to target length."""
        if tensor.shape[1] >= target_length:
            return tensor
        padding = torch.zeros(
            tensor.shape[0],
            target_length - tensor.shape[1],
            tensor.shape[2],
            dtype=tensor.dtype
        )
        return torch.cat([tensor, padding], dim=1)
