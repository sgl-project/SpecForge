# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from typing import Optional, Union, List

from torch import nn
from torchtune.models.clip._component_builders import (
    clip_vision_encoder,
)
# from torchtune.models.llama4._chunked_attention import get_chunked_attention_mask
from spec.models.llama4._eagle_attention import create_ttt_mask

from torchtune.models.llama4._encoder import (
    Llama4VisionEncoder,
    Llama4VisionProjectionHead,
)
from torchtune.models.llama4._position_embeddings import Llama4ScaledRoPE
from torchtune.modules import (
    FeedForward,
    FrozenNF4Linear,
    MultiHeadAttention,
    rms_norm,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    TransformerDraftAttentionLayer
)
from torchtune.modules.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)

"""
Component builders for the Llama4 model.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def llama4_draft_decoder(
    *,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    hidden_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    norm_eps: float = 1e-5,
    use_qk_norm: bool = True,
    skip_rope_interval: Optional[int] = None,
    attention_chunk_size: Optional[int] = None,
) -> TransformerDecoder:

    if skip_rope_interval is not None and attention_chunk_size is None:
        raise ValueError(
            "Must pass local_chunk_size when enabling local chunked attention"
        )
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    layers = []
    for i in range(num_layers):

        mask_mod = None
        if skip_rope_interval is not None and (i + 1) % skip_rope_interval != 0:
            mask_mod = partial(
                create_ttt_mask, chunk_size=attention_chunk_size
            )
            # Note: this is the value in llama-models, which doesn't match the config
            pos_embeddings = rope

            q_norm = partial(rms_norm, eps=norm_eps) if use_qk_norm else None
            k_norm = partial(rms_norm, eps=norm_eps) if use_qk_norm else None
        else:
            pos_embeddings, q_norm, k_norm = None, None, None

        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim * 2, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim * 2, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim * 2, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=pos_embeddings,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        
        mlp_layer = llama4_mlp(dim=embed_dim, hidden_dim=hidden_dim)

        layer = TransformerDraftAttentionLayer(
            attn=self_attn,
            mlp=mlp_layer,
            sa_norm=None,
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mask_mod=mask_mod,
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Identity()
    output_proj = nn.Identity
    
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        output=output_proj
    )


class EAGLE3DraftModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        num_heads,
        num_kv_heads,
        embed_dim,
        intermediate_dim,
        max_seq_len,
        rope_base,
        norm_eps,
        num_feature_layers,
        dropout
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.num_feature_layers = num_feature_layers
        self.dropout = dropout

        # 1. fusion low mid high hidden size
        self.feature_fusion = nn.Linear(
            self.num_feature_layers * self.embed_dim,
            self.embed_dim,
            bias=False
        )

        # 2. input norm
        self.input_embeds_norm = RMSNorm(dim=self.embed_dim, eps=self.norm_eps)
        self.fused_features_norm = RMSNorm(dim=self.embed_dim, eps=self.norm_eps)

        # 3. Draft decoder (one layer for base model, but using mlp rather than moe)
        self.draft_decoder = llama4_draft_decoder(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            embed_dim=self.embed_dim,
            hidden_dim=self.intermediate_dim,
            max_seq_len=self.max_seq_len,
            rope_base=self.rope_base,
            norm_eps=self.norm_eps,
            attn_dropout=self.dropout,
        )

    def to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        self.feature_fusion.to_empty(device=device, recurse=recurse)
        self.input_embeds_norm.to_empty(device=device, recurse=recurse)
        self.fused_features_norm.to_empty(device=device, recurse=recurse)
        
        self.draft_decoder.norm.to_empty(device=device, recurse=recurse)
        
        for layer in self.draft_decoder.layers:
        
            layer.mlp_norm.to_empty(device=device, recurse=recurse)
        
        
            layer.attn.q_proj.to_empty(device=device, recurse=recurse)
            layer.attn.k_proj.to_empty(device=device, recurse=recurse)
            layer.attn.v_proj.to_empty(device=device, recurse=recurse)
            layer.attn.output_proj.to_empty(device=device, recurse=recurse)
            
            layer.mlp.w1.to_empty(device=device, recurse=recurse)
            layer.mlp.w2.to_empty(device=device, recurse=recurse)
            layer.mlp.w3.to_empty(device=device, recurse=recurse)



    def initialize_parameters(self):
        """Initialize weights for the draft model."""
        # Initialize feature fusion layer
        nn.init.xavier_uniform_(self.feature_fusion.weight)
        if self.feature_fusion.bias is not None:
            nn.init.zeros_(self.feature_fusion.bias)

        # Initialize RMSNorm layers
        nn.init.ones_(self.input_embeds_norm.scale)
        nn.init.ones_(self.fused_features_norm.scale)

        nn.init.ones_(self.draft_decoder.norm.scale)

        # Initialize draft decoder layers
        for layer in self.draft_decoder.layers:
            layer.attn.q_proj.reset_parameters()
            layer.attn.k_proj.reset_parameters()
            layer.attn.v_proj.reset_parameters()
            layer.attn.output_proj.reset_parameters()
            
            nn.init.ones_(layer.mlp_norm.scale)
            
            layer.mlp.w1.reset_parameters()
            layer.mlp.w2.reset_parameters()
            layer.mlp.w3.reset_parameters()
            

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        input_hidden: Optional[List[torch.Tensor]] = None,
    ):
        feature_list = []
        for feature in input_hidden:
            feature_list.append(feature)
 
        concatenated_features = torch.cat(feature_list, dim=-1)
        fused_features = self.feature_fusion(concatenated_features)

        input_embeds = self.input_embeds_norm(input_embeds)
        fused_features = self.fused_features_norm(fused_features)
        
        # [B, seq_len, 2*embed_dim] -> [B, seq_len, embed_dim]
        combined_input = torch.cat([input_embeds, fused_features], dim=-1)
        # projected_input = self.input_projection(combined_input)

        draft_outputs = self.draft_decoder(
            tokens=None,
            mask=mask,
            input_embeds=combined_input,
        )

        hidden_states = draft_outputs

        return hidden_states


def llama4_draft_model(
    *,
    vocab_size: int = 128256,
    num_layers: int = 1,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    embed_dim: int = 4096,
    intermediate_dim: int = 11008,
    max_seq_len: int = 8192,
    rope_base: float = 500000,
    norm_eps: float = 1e-5,
    num_feature_layers: int = 3,
    dropout: float = 0.0,
) -> nn.Module:
    """
    EAGLE-3 Draft Model Builder
    
    Args:
        vocab_size: Vocabulary size
        num_layers: Number of transformer layers (usually 1 for draft model)  
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        embed_dim: Hidden dimension
        intermediate_dim: MLP intermediate dimension
        max_seq_len: Maximum sequence length
        rope_base: RoPE base frequency
        norm_eps: Layer norm epsilon
        num_feature_layers: Number of target model layers to fuse (low, mid, high)
        dropout: Dropout rate
        
    Returns:
        EAGLE3DraftModel instance
    """

    return EAGLE3DraftModel(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        norm_eps=norm_eps,
        num_feature_layers=num_feature_layers,
        dropout=dropout,
    )


def llama4_vision_encoder(
    # clip encoder parameters
    *,
    patch_size: int,
    num_heads: int,
    clip_embed_dim: int,
    clip_num_layers: int,
    clip_hidden_states: Optional[list[int]] = None,
    # projection parameters
    projection_embed_dim: int,
    decoder_embed_dim: int,
    # image parameters
    tile_size: int,
    max_num_tiles: int = 1,
    in_channels: int = 3,
) -> Llama4VisionEncoder:
    """
    Build the Llama 4 vision encoder by combining the CLIP image model with an additional
    projection head fusion module. This includes:
    - Spatial positional encodings
    - CLIP model backbone
    - MLP head projecting CLIP outputs into embeddings for the decoder

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        num_heads (int): The number of attention heads in each transformer layer.
        clip_embed_dim (int): The dimensionality of each patch embedding in CLIP.
        clip_num_layers (int): The number of transformer layers.
        clip_hidden_states (Optional[list[int]]): The indices of CLIP hidden layers to return
            to return to the encoder projection head. It will return the intermediate results
            of the vision transformer layers which will be concatenated with the CLIP output
            and input into the projection head. For example, ``clip_hidden_states=[0,3]`` will
            return the embeddings before they go through the first and fourth layers.
        projection_embed_dim (int): The dimensionality of the linear layers in the projection head.
        decoder_embed_dim (int): The dimensionality of the final output embeddings for the decoder.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings. Default is 1, treating the image as a single tile.
        in_channels (int): The number of image input channels.

    Returns:
        Llama4VisionEncoder: Instantiation of Llama 4 vision encoder.
    """

    # clip encoder
    clip = clip_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=clip_embed_dim,
        num_layers=clip_num_layers,
        num_heads=num_heads,
        activation=nn.GELU,
        out_indices=clip_hidden_states,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        attn_bias=True,
        output_cls_projection=False,
        append_cls_token=True,
        use_rope=True,
        use_tile_pos_embed=False,
    )

    # Projection head
    projection_head = llama4_vision_projection_head(
        decoder_embed_dim=decoder_embed_dim,
        clip_embed_dim=clip_embed_dim,
        projection_embed_dim=projection_embed_dim,
    )

    return Llama4VisionEncoder(clip=clip, projection_head=projection_head)


def llama4_vision_projection_head(
    *,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    projection_embed_dim: int,
) -> Llama4VisionProjectionHead:
    """
    Build the Llama 4 Vision Projection Head that maps the output of the CLIP encoder
    to embeddings that can be fed into the decoder.

    Args:
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        projection_embed_dim (int): embedding dimension for the inner linear layers in the projection head.

    Returns:
        Llama4VisionProjectionHead: Instantiation of Llama 4 vision projection head.
    """
    pixel_shuffle_scaling_factor = 0.5
    output = nn.Sequential(
        nn.Linear(
            # Account for the pixel shuffle scaling factor ** 2
            int(clip_embed_dim // (pixel_shuffle_scaling_factor**2)),
            projection_embed_dim,
            bias=False,
        ),
        nn.GELU(),
        nn.Linear(projection_embed_dim, projection_embed_dim, bias=False),
        nn.GELU(),
        nn.Linear(projection_embed_dim, decoder_embed_dim, bias=False),
    )

    return Llama4VisionProjectionHead(
        output=output,
        pixel_shuffle_scaling_factor=pixel_shuffle_scaling_factor,
    )


def llama4_decoder(
    *,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    hidden_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    norm_eps: float = 1e-5,
    num_experts: int = 16,
    experts_per_token: int = 1,
    use_shared_expert: bool = True,
    use_qk_norm: bool = True,
    moe_every_n_layers: Optional[int] = None,
    mlp_hidden_dim: Optional[int] = None,
    skip_rope_interval: Optional[int] = None,
    attention_chunk_size: Optional[int] = None,
    use_scaled_rope: bool = False,
    rope_scale_factor: Optional[float] = 16.0,
    rope_low_freq_factor: Optional[float] = 1.0,
    rope_high_freq_factor: Optional[float] = 1.0,
    old_context_len: Optional[int] = 8192,
) -> TransformerDecoder:
    """
    Build the decoder associated with the MOE model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        hidden_dim (int): hidden dimension for the MoeLayer
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        norm_eps (float): epsilon in RMS norms. Default: 1e-5
        num_experts (int): Number of experts in each moe layer. Default: 16
        experts_per_token (int): Number of experts each token will choose in Token Choice. Default: 2
        use_shared_expert (bool): Whether to use a shared expert or not. Default: True
        use_qk_norm (bool): Whether to use qk norm in RoPE layers. Default: True
        moe_every_n_layers (Optional[int]): Frequency of inserting MoE layers in the decoder.
            If set, every nth layer will be an MoE layer. Default: MoE every layer
        mlp_hidden_dim (Optional[int]): Hidden dim for any MLP (i.e. non-MoE) layers.
            Only applicable if moe_every_n_layers is not None.
        skip_rope_interval (Optional[int]): Frequency of inserting local attention layers in the decoder.
            If set, every nth layer will use local attention. Default is to always use vanilla attention
        attention_chunk_size (Optional[int]): Size of chunks for local attention.
            Required if `skip_rope_interval` is set.
        use_scaled_rope (bool): Whether to use scaled RoPE or not. Scaled RoPE is used for Llama4 Scout
            model, but not Maverick model. Default: False
        rope_scale_factor (Optional[float]): scaling factor for RoPE. Only applicable if use_scaled_rope=True.
            Default: 16.0
        rope_low_freq_factor (Optional[float]): scaling factor for low frequency RoPE. Only applicable if
            use_scaled_rope=True. Default: 1.0
        rope_high_freq_factor (Optional[float]): scaling factor for high frequency RoPE. Only applicable if
            use_scaled_rope=True. Default: 1.0
        old_context_len (Optional[int]): old context length for scaling theta. Only applicable if
            use_scaled_rope=True. Default: 8192

    Returns:
        TransformerDecoder: Instantiation of MoE model.
    """
    if skip_rope_interval is not None and attention_chunk_size is None:
        raise ValueError(
            "Must pass local_chunk_size when enabling local chunked attention"
        )
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    if use_scaled_rope:
        rope = Llama4ScaledRoPE(
            dim=head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
            scale_factor=rope_scale_factor,
            low_freq_factor=rope_low_freq_factor,
            high_freq_factor=rope_high_freq_factor,
            old_context_len=old_context_len,
        )
    else:
        rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=max_seq_len, base=rope_base
        )
    layers = []
    for i in range(num_layers):

        mask_mod = None
        if skip_rope_interval is not None and (i + 1) % skip_rope_interval != 0:
            mask_mod = partial(
                get_chunked_attention_mask, chunk_size=attention_chunk_size
            )
            # Note: this is the value in llama-models, which doesn't match the config
            pos_embeddings = rope

            q_norm = partial(rms_norm, eps=norm_eps) if use_qk_norm else None
            k_norm = partial(rms_norm, eps=norm_eps) if use_qk_norm else None
        else:
            pos_embeddings, q_norm, k_norm = None, None, None

        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=pos_embeddings,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        is_moe = moe_every_n_layers is None or (i + 1) % moe_every_n_layers == 0
        if is_moe:
            mlp_layer = llama4_moe(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                experts_per_token=experts_per_token,
                use_shared_expert=use_shared_expert,
            )
        else:
            mlp_layer = llama4_mlp(dim=embed_dim, hidden_dim=mlp_hidden_dim)

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp_layer,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mask_mod=mask_mod,
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)
    
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    
    num_layers = len(layers)
    low = num_layers // 4
    mid = num_layers // 2 - 2
    high = num_layers - 2
    output_hidden_states = [low, mid, high]
    
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        output=output_proj,
        output_hidden_states=output_hidden_states
    )


def llama4_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    down_proj = (
        nn.Linear(hidden_dim, dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(hidden_dim, dim, bias=False)
    )
    up_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


def llama4_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int = 8,
    experts_per_token: int = 1,
    use_shared_expert: bool = True,
) -> MoE:
    """
    Build the MoE layer associated with the Llama model.

    Args:
        dim (int): Input dimension of experts.
        hidden_dim (int): Hidden dimension of experts.
        num_experts (int): Number of experts in each MoE layer. Default: 8
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
        use_shared_expert (bool): Whether to use a shared expert or not. Default: True

    Returns:
        MoE: Instantiation of MoE layer.
    """
    router = TokenChoiceTopKRouter(
        gate=nn.Linear(dim, num_experts, bias=False),
        dim=dim,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
    )
    experts = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    shared_expert = (
        llama4_mlp(dim=dim, hidden_dim=hidden_dim) if use_shared_expert else None
    )
    return MoE(
        experts=experts,
        router=router,
        shared_expert=shared_expert,
    )

