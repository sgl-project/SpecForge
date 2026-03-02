# coding=utf-8
"""Single-Model Speculative Decoding Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineSharedBackendModel(nn.Module):
    """Training wrapper for single-model speculative decoding.

    This model implements a training approach that uses:
    1. Cross-entropy loss between draft logits and target token distribution
    2. MSE loss between draft layer K/V and corresponding target layer K/V

    The combined loss encourages the draft model to:
    - Predict the same tokens as the target model (CE loss)
    - Learn similar internal representations (MSE loss)

    Args:
        model: The Qwen3SharedDraftModel to train
        block_size: Block size for parallel training (default: 16)
        ce_weight: Weight for cross-entropy loss (default: 1.0)
        mse_weight: Weight for MSE loss between K/V pairs (default: 0.1)
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 16,
        ce_weight: float = 1.0,
        mse_weight: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.block_size = block_size
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass with combined loss computation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask
            loss_mask: Mask for which positions contribute to loss
            position_ids: Position IDs
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - loss: Combined loss (ce_weight * ce_loss + mse_weight * mse_loss)
                - ce_loss: Cross-entropy loss component
                - mse_loss: MSE loss component
                - draft_logits: Logits from the draft model
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

        # Get target logits from the frozen target model
        with torch.no_grad():
            target_outputs = self.model.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            target_logits = target_outputs.logits

        draft_logits = outputs.logits

        # Compute cross-entropy loss
        # Use target's argmax as the training target (knowledge distillation style)
        vocab_size = draft_logits.shape[-1]
        batch_size, seq_len, _ = draft_logits.shape

        # Flatten for loss computation
        draft_logits_flat = draft_logits.view(-1, vocab_size)
        target_logits_flat = target_logits.view(-1, vocab_size)

        # Create targets from target logits (hard targets)
        targets = target_logits_flat.argmax(dim=-1)

        # Compute CE loss per token
        ce_loss_per_token = F.cross_entropy(
            draft_logits_flat,
            targets,
            reduction='none',
        )

        # Apply loss mask if provided
        if loss_mask is not None:
            loss_mask_flat = loss_mask.view(-1)
            ce_loss = (ce_loss_per_token * loss_mask_flat).sum() / (loss_mask_flat.sum() + 1e-6)
        else:
            ce_loss = ce_loss_per_token.mean()

        # Compute MSE loss between draft and target representations
        # We need to extract K/V from the target layers and compare with draft K/V
        mse_loss = self._compute_kv_mse_loss(input_ids, attention_mask, position_ids)

        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.mse_weight * mse_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mse_loss": mse_loss,
            "draft_logits": draft_logits,
            "target_logits": target_logits,
        }

    def _compute_kv_mse_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute MSE loss between draft and target K/V pairs.

        This extracts K/V from both the draft layers and their corresponding
        target layers, then computes the MSE loss.

        Returns:
            Average MSE loss across all draft layers
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Get target hidden states
        with torch.no_grad():
            target_outputs = self.model.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        # Get embeddings
        hidden_states = target_outputs.hidden_states[0]
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Compute MSE loss for each draft layer
        mse_loss = 0.0
        num_layers = 0

        for i, layer in enumerate(self.model.layers):
            target_layer_id = self.model.target_layer_ids[i]

            # Extract target K/V
            target_hidden = target_outputs.hidden_states[target_layer_id + 1]
            target_kv = self.model.extract_target_kv(target_hidden, target_layer_id)

            # Run draft layer to get draft K/V
            with torch.enable_grad():
                # For first layer, use embeddings; for subsequent layers, we'd need to chain
                # For simplicity, we compute a single-step MSE here
                _, draft_kv = layer(
                    hidden_states=hidden_states,
                    target_kv=target_kv,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            # Compute MSE loss for this layer
            # We compare the draft K/V with the target K/V
            # Note: shapes may differ due to bi-directional attention concatenation
            # We compare only the draft portion
            target_k, target_v = target_kv
            draft_k, draft_v = draft_kv

            # The draft K/V includes both context (target) and draft portions
            # We compare the draft portion (second half of K/V)
            draft_len = draft_k.shape[1] - target_k.shape[1]

            if draft_len > 0:
                # Extract draft portion
                draft_k_draft = draft_k[:, -draft_len:, :, :]
                draft_v_draft = draft_v[:, -draft_len:, :, :]

                # Pad or truncate to match shapes for MSE computation
                # For simplicity, compute MSE on matching sequence positions
                min_len = min(target_k.shape[1], draft_k_draft.shape[1])

                if min_len > 0:
                    layer_mse = (
                        F.mse_loss(
                            draft_k_draft[:, :min_len, :, :],
                            target_k[:, :min_len, :, :],
                        )
                        + F.mse_loss(
                            draft_v_draft[:, :min_len, :, :],
                            target_v[:, :min_len, :, :],
                        )
                    ) / 2.0
                    mse_loss += layer_mse
                    num_layers += 1

            # Update hidden states for next layer (simplified - in practice would chain)
            # For proper MSE computation, we'd need to run through all layers
            # This is a simplified version for demonstration
            if i == 0:
                with torch.enable_grad():
                    hidden_states, _ = layer(
                        hidden_states=hidden_states,
                        target_kv=target_kv,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

        if num_layers > 0:
            mse_loss = mse_loss / num_layers

        return mse_loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens using the draft model for verification.

        This is a simplified generation method for testing purposes.
        In production, use the full speculative decoding pipeline.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0 for greedy)
            **kwargs: Additional arguments

        Returns:
            Generated token IDs
        """
        self.model.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Get target model for final verification
        target_model = self.model.target_model

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get draft predictions
            draft_outputs = self.model(generated)
            draft_logits = draft_outputs.logits[:, -1, :]

            if temperature < 1e-5:
                draft_token = draft_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(draft_logits / temperature, dim=-1)
                draft_token = torch.multinomial(probs, num_samples=1)

            # Verify with target model
            target_outputs = target_model(generated)
            target_logits = target_outputs.logits[:, -1, :]

            if temperature < 1e-5:
                target_token = target_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(target_logits / temperature, dim=-1)
                target_token = torch.multinomial(probs, num_samples=1)

            # Accept draft token if it matches, otherwise use target token
            accepted = (draft_token == target_token)
            generated = torch.cat([generated, torch.where(accepted, draft_token, target_token)], dim=1)

        return generated
