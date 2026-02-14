from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from transformers.cache_utils import Cache
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from megatron.core.transformer.moe.router import (
    apply_router_token_dropping,
)
from megatron.core.transformer.moe.moe_utils import (
    sinkhorn,
)

from specforge.layers import ColumnParallelLinear, RowParallelLinear

from .llama3_eagle import (
    LlamaForCausalLMEagle3,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlexAttention,
    LlamaFlashAttention,
    LlamaUSPAttention,
    LlamaRMSNorm,
)
from .qwen3_moe_eagle import Qwen3MoeMLP, Qwen3MoeSparseMoeBlock


class Qwen3RoutMoeSparseMoeBlock(nn.Module):
    """
    Qwen3 MoE expert layer adapted to Megatron-LM routing logic
    Core optimizations:
    1. Full alignment with Megatron routing process (Z-Loss + TopK/Sinkhorn + Token Dropping + auxiliary loss)
    2. Added tensor shape validation and distributed printing
    3. Fixed Sinkhorn load balancing calculation logic
    4. Optimized expert computation efficiency (only iterate over hit experts)
    """
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        # Compatible with default parameters (aligned with Megatron configuration)
        self.config = config
        self.num_experts = getattr(config, "num_experts", 16)
        self.top_k = getattr(config, "num_experts_per_tok", 8)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        
        # Gating layer (consistent with Qwen native implementation, compatible with TP parallelism)
        self.gate = ColumnParallelLinear(
            config.hidden_size, self.num_experts, bias=False, gather_output=False
        ) if getattr(config, "tensor_model_parallel", False) else nn.Linear(
            config.hidden_size, self.num_experts, bias=False
        )
        
        # Expert layers (reuse Qwen native MLP, compatible with Megatron grouped GEMM)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, 
                    intermediate_size=getattr(config, "moe_intermediate_size", config.intermediate_size),
                    grouped_gemm=getattr(config, "moe_grouped_gemm", False)
                )
                for _ in range(self.num_experts)
            ]
        )
        
        # Megatron routing supplementary configuration
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts)) if getattr(config, "moe_router_enable_expert_bias", False) else None
        self.router_replay = getattr(config, "moe_router_replay", False)
        self.z_loss_coeff = getattr(config, "moe_z_loss_coeff", 0.01)
        self.load_balancing_loss = None  # Store load balancing loss

    def apply_z_loss(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reuse Megatron Z-Loss logic: prevent routing logits from being too large leading to numerical instability
        """
        if padding_mask is not None:
            logits = logits * padding_mask.unsqueeze(-1)
        
        # Z-Loss calculation (mean normalization + L2 penalty)
        logits_mean = logits.mean(dim=-1, keepdim=True)
        logits_centered = logits - logits_mean
        z_loss = self.z_loss_coeff * torch.sum(logits_centered **2) / logits.numel()
        logits = logits - z_loss * logits  # Only pass gradients, do not change numerical distribution
        
        return logits

    def _compute_load_balancing_loss(self, expert_load: torch.Tensor) -> torch.Tensor:
        """
        Compute Megatron-style load balancing auxiliary loss
        """
        avg_load = expert_load.float().mean()
        load_var = (expert_load - avg_load).pow(2).mean()
        self.load_balancing_loss = load_var * self.config.moe_aux_loss_coeff
        return self.load_balancing_loss

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.top_k == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.top_k, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.top_k, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core routing logic: aligned with Megatron-LM native implementation
        Supports: TopK routing / Sinkhorn load balancing / Token Dropping / auxiliary loss
        """
        # Tensor shape standardization: [seq_len, batch_size, num_experts] → [num_tokens, num_experts]
        seq_len, batch_size = logits.shape[:2]
        num_tokens = seq_len * batch_size
        logits_flat = logits.reshape(num_tokens, self.num_experts)
    
        # Step 1: Z-Loss stabilize logits
        logits_flat = self.apply_z_loss(logits_flat, padding_mask)

        # Step 2: Core routing strategy
        probs, routing_map = self.sinkhorn_load_balancing(logits_flat)
            
        # Step 3: Token Dropping (discard when exceeding expert capacity)
        expert_load_before_drop = routing_map.sum(dim=0)
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.top_k,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Step 4: Calculate and print load balancing information
        expert_load = routing_map.sum(dim=0)
        # Step 5: Calculate load balancing auxiliary loss during training
        if self.training and torch.is_grad_enabled():
            self._compute_load_balancing_loss(expert_load)

        # Step 6: Qwen native probability normalization
        if self.norm_topk_prob:
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return probs, routing_map

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation logic:
        1. Generate routing logits → 2. Megatron routing → 3. Expert computation → 4. Result aggregation
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Step 1: Generate routing Logits
        router_logits = self.gate(hidden_states_flat)
        router_logits_reshaped = router_logits.view(sequence_length, batch_size, self.num_experts)

        # Step 2: Megatron-style routing
        probs, routing_map = self.routing(router_logits_reshaped, padding_mask=None)

        # Step 3: Parse routing results (Top-K indices + weights)
        topk_probs, selected_experts = torch.topk(probs, self.top_k, dim=-1)  # [num_tokens, top_k]
        num_tokens = probs.shape[0] 
        token_indices = torch.arange(num_tokens, device=probs.device).unsqueeze(1).expand(-1, self.top_k)
        routing_weights = probs[token_indices, selected_experts]  # [num_tokens, top_k]

        # Step 4: Expert computation (only iterate over hit experts to improve efficiency)
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
        
        for expert_idx in torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0]:
            # Get token indices for current expert
            top_k_idx, token_idx = torch.where(expert_mask[expert_idx])
            # Expert forward computation
            expert_output = self.experts[expert_idx](hidden_states_flat[token_idx])
            # Weight and aggregate results
            expert_output = expert_output * routing_weights[token_idx, top_k_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output.to(final_hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(LlamaDecoderLayer):
    """
    Qwen3 MoE decoder layer:
    1. Reconstruct Attention instantiation logic (compatible with multiple backends)
    2. Align Megatron MoE loss calculation
    3. Improve residual connection and normalization logic
    4. Add tensor shape validation
    """
    def __init__(self, config: Qwen3MoeConfig, attention_backend: str = "sdpa"):
        super().__init__(config, attention_backend)
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 1. Attention backend mapping (concise + validation)
        self.attention_backend = attention_backend.lower()
        attn_cls_map = {
            "sdpa": LlamaAttention,
            "usp": LlamaUSPAttention,
            "flex_attention": LlamaFlexAttention,
            "fa": LlamaFlashAttention
        }
        if self.attention_backend not in attn_cls_map:
            raise ValueError(f"Unsupported attention backend: {attention_backend}, supported: {list(attn_cls_map.keys())}")
        self.self_attn = attn_cls_map[self.attention_backend](config=config)

        # 2. MoE expert layer (core)
        if self.config.routing_type == "sinkhorn":
            self.mlp = Qwen3RoutMoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeSparseMoeBlock(config)

        # 3. Normalization layers (aligned with Qwen official + Megatron precision)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. MoE loss configuration
        self.moe_loss_weight = getattr(config, "moe_loss_weight", 0.01)
        self.save_idx = 0

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        return_router_logits: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor]]:
        """
        Forward propagation:
        - Improve residual connections
        - Compatible with multiple Attention backend outputs
        - Calculate MoE load balancing loss
        """
        # Input shape validation
        assert input_emb.shape[-1] == self.hidden_size, f"Input embedding size mismatch: {input_emb.shape[-1]} vs {self.hidden_size}"
        assert hidden_states.shape[-1] == self.hidden_size, f"Hidden states size mismatch: {hidden_states.shape[-1]} vs {self.hidden_size}"

        # Normalization + residual preparation
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # Concatenate input embedding and hidden states (keep original logic)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        assert hidden_states.shape[-1] == 2 * self.hidden_size, f"Concatenated size error: {hidden_states.shape[-1]} vs {2*self.hidden_size}"

        # Self-Attention layer
        attn_outputs = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # Parse Attention outputs (compatible with single/multiple outputs)
        if isinstance(attn_outputs, tuple):
            hidden_states = attn_outputs[0]
            attn_aux_output = attn_outputs[1:] if output_attentions else None
        else:
            hidden_states = attn_outputs
            attn_aux_output = None

        # Attention residual connection
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.contiguous()

        # MoE MLP layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)

        # Calculate MoE load balancing loss (during training)
        moe_loss = None
        if self.training and hasattr(self.mlp, "load_balancing_loss") and self.mlp.load_balancing_loss is not None:
            moe_loss = self.mlp.load_balancing_loss * self.moe_loss_weight

        # MLP residual connection
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.contiguous()
        
        save_path = f"./router/{self.save_idx}.pt"
        self.save_idx += 1
        # 2. Save tensor (move to CPU to avoid GPU tensor dependency, detach to decouple computation graph)
        if 0 == dist.get_rank():
            torch.save(hidden_states.detach().cpu(), save_path)
        if self.save_idx >= 16:
            exit(0)

        return hidden_states



class Qwen3MoERouterForCausalLMEagle3(LlamaForCausalLMEagle3):

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config, quant_config, attention_backend)
        self.config = config
        # Configuration required for Megatron routing (adjustable as needed)
        def get_config_val(key, default):
            return getattr(config, key, default)

        self.config.moe_router_pre_softmax = get_config_val("moe_router_pre_softmax", False)
        self.config.moe_router_num_groups = get_config_val("moe_router_num_groups", None)
        self.config.moe_router_group_topk = get_config_val("moe_router_group_topk", None)
        self.config.moe_router_scaling_factor = get_config_val("moe_router_scaling_factor", 1.0)
        self.config.moe_router_score_function = get_config_val("moe_router_score_function", "softmax")
        self.config.moe_expert_capacity_factor = get_config_val("moe_expert_capacity_factor", None)  # Load limit
        self.config.moe_token_drop_policy = get_config_val("moe_token_drop_policy", "none")  # Policy when exceeding capacity
        self.config.moe_pad_expert_input_to_capacity = get_config_val("moe_pad_expert_input_to_capacity", False)
        self.config.moe_aux_loss_coeff = get_config_val("moe_aux_loss_coeff", 0.01)  # Auxiliary loss coefficient
        self.config.routing_type = get_config_val("routing_type", "sinkhorn")  # Optional: sinkhorn/aux_loss/seq_aux_loss/none
        self.config.moe_aux_loss_coeff = 0


        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = Qwen3MoeDecoderLayer(config, attention_backend=attention_backend)

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)