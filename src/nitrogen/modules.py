from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from pydantic import BaseModel, Field
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.compute_dtype = compute_dtype

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embeds timesteps.
        Args:
            timesteps: (N,) tensor of timesteps.
        Returns:
            (N, D) tensor of embeddings.
        """
        timesteps_proj = self.time_proj(timesteps).to(self.compute_dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(
            embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_type = norm_type

        # Positional Embeddings
        if positional_embeddings == "sinusoidal":
            if num_positional_embeddings is None:
                raise ValueError(
                    "num_positional_embeddings must be defined for sinusoidal embeddings."
                )
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # 1. Norm 1 & Self-Attention
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, norm_eps=norm_eps)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Norm 3 & Feed-Forward
        self.norm3 = nn.LayerNorm(
            dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention Block
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2. Feed-Forward Block
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class DiTConfig(BaseModel):
    num_attention_heads: int = 8
    attention_head_dim: int = 64
    output_dim: int = 26
    num_layers: int = 12
    dropout: float = 0.1
    attention_bias: bool = True
    activation_fn: str = "gelu-approximate"
    upcast_attention: bool = False
    norm_type: str = "ada_norm"
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-5
    max_num_positional_embeddings: int = 512
    compute_dtype: str = "float32"
    final_dropout: bool = True
    positional_embeddings: Optional[str] = "sinusoidal"
    interleave_self_attention: bool = False
    cross_attention_dim: Optional[int] = Field(
        default=None,
        description="Dimension of the cross-attention embeddings. If None, no cross-attention is used.",
    )


class DiT(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.compute_dtype = getattr(torch, self.config.compute_dtype)
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=self.compute_dtype
        )

        blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and self.config.interleave_self_attention
            curr_cross_attention_dim = (
                self.config.cross_attention_dim if not use_self_attn else None
            )

            blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=self.config.norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=self.config.positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=self.config.final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )

        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformerConfig(BaseModel):
    num_attention_heads: int = 8
    attention_head_dim: int = 64
    num_layers: int = 12
    dropout: float = 0.1
    attention_bias: bool = True
    activation_fn: str = "gelu-approximate"
    upcast_attention: bool = False
    max_num_positional_embeddings: int = 512
    final_dropout: bool = True
    positional_embeddings: Optional[str] = "sinusoidal"


class SelfAttentionTransformer(ModelMixin):
    _supports_gradient_checkpointing = True

    def __init__(self, config: SelfAttentionTransformerConfig):
        super().__init__()
        self.config = config

        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=self.config.positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=self.config.final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_hidden_states: bool = False,
    ):
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
