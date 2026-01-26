# Copyright 2024 SpecForge Authors
# Licensed under the Apache License, Version 2.0

"""
VLM Dataset wrapper for SpecForge.

This module provides the VLMDataset class that wraps a HuggingFace dataset
and uses a VLMTemplate to encode samples on-the-fly during DataLoader iteration.
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .vlm_template.base import VLMTemplate


class VLMDataset(Dataset):
    """
    Dataset wrapper that applies VLM template encoding on-the-fly.
    
    This design enables:
    - dataset.map() to only do basic formatting (multiprocessing-safe)
    - Image processing happens in main process during DataLoader iteration
    - Template-specific encoding logic is encapsulated
    
    Usage:
        template = MiniCPMVTemplate(processor_path, max_length)
        dataset = VLMDataset(hf_dataset, template)
        dataloader = DataLoader(dataset, collate_fn=template.collate)
    """
    
    def __init__(
        self,
        hf_dataset,
        template: VLMTemplate,
        skip_failed: bool = True,
    ):
        """
        Initialize VLMDataset.
        
        Args:
            hf_dataset: HuggingFace dataset with 'image' and 'conversations' columns
            template: VLM template for encoding
            skip_failed: If True, return None for failed samples (filtered in collate)
        """
        self.dataset = hf_dataset
        self.template = template
        self.skip_failed = skip_failed
        self._valid_indices = None
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get and encode a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Encoded sample dictionary, or None if encoding fails
        """
        example = self.dataset[idx]
        
        try:
            encoded = self.template.encode(example)
            if encoded is None and not self.skip_failed:
                logging.warning(f"Sample {idx} encoding returned None")
            return encoded
        except Exception as e:
            if self.skip_failed:
                logging.warning(f"Failed to encode sample {idx}: {e}")
                return None
            raise


class VLMIterableDataset(torch.utils.data.IterableDataset):
    """
    Iterable dataset wrapper for streaming VLM data.
    
    Useful for very large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        hf_dataset,
        template: VLMTemplate,
        skip_failed: bool = True,
    ):
        self.dataset = hf_dataset
        self.template = template
        self.skip_failed = skip_failed
    
    def __iter__(self):
        for example in self.dataset:
            try:
                encoded = self.template.encode(example)
                if encoded is not None:
                    yield encoded
            except Exception as e:
                if not self.skip_failed:
                    raise
                logging.warning(f"Failed to encode sample: {e}")
                continue


def create_vlm_dataset(
    hf_dataset,
    template: VLMTemplate,
    streaming: bool = False,
    skip_failed: bool = True,
) -> Dataset:
    """
    Create a VLM dataset wrapper.
    
    Args:
        hf_dataset: HuggingFace dataset
        template: VLM template for encoding
        streaming: If True, use iterable dataset for streaming
        skip_failed: If True, skip failed samples
    
    Returns:
        VLMDataset or VLMIterableDataset
    """
    if streaming:
        return VLMIterableDataset(hf_dataset, template, skip_failed)
    return VLMDataset(hf_dataset, template, skip_failed)
