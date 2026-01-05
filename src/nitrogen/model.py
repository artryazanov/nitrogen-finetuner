from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from einops import rearrange
from pydantic import BaseModel, Field
from torch import nn
from torch.distributions import Beta
from transformers import AutoModel, SiglipVisionModel

from src.nitrogen.modules import (
    DiT,
    DiTConfig,
    SelfAttentionTransformer,
    SelfAttentionTransformerConfig,
)


class NitroGenConfig(BaseModel):
    model_type: str = Field(default="nitrogen", frozen=True)

    add_pos_embed: bool = Field(
        default=False, description="Whether to add positional embedding"
    )
    model_dtype: str = Field(default="float32", description="Model data type.")
    diffusion_model_cfg: DiTConfig = Field(
        ..., description="Diffusion model configuration."
    )
    vl_self_attention_cfg: SelfAttentionTransformerConfig = Field(
        ..., description="VL self-attention configuration."
    )
    hidden_size: int = Field(default=1024, description="Input embedding dimension.")
    max_seq_len: int = Field(default=1024, description="Maximum Sequence Length")
    action_dim: int = Field(default=None, description="Action dimension.")
    action_horizon: int = Field(default=None, description="Action horizon.")
    noise_beta_alpha: float = Field(default=1.5, description="")
    noise_beta_beta: float = Field(default=1.0, description="")
    noise_s: float = Field(
        default=0.999, description="Flow matching noise Beta distribution s."
    )
    num_timestep_buckets: int = Field(
        default=1000, description="Number of timestep discretization buckets."
    )
    num_inference_timesteps: int = Field(
        default=None, description="Number of inference steps for noise diffusion."
    )
    max_num_embodiments: int = Field(default=1, description="Number of embodiments.")
    vision_encoder_name: str = Field(
        default="google/siglip-large-patch16-256", description="Vision encoder name."
    )
    vision_hidden_size: int = Field(default=768, description="Siglip hidden size.")
    add_view_embed: bool = Field(
        default=False, description="Whether to add view embedding."
    )

    tune_vision_tower: bool = Field(default=True, description="Tune vision if True.")
    tune_mm_projector: bool = Field(
        default=True, description="Tune mm projector if True."
    )
    tune_diffusion_model: bool = Field(
        default=True, description="Tune diffusion model if True."
    )
    tune_multi_projector: bool = Field(
        default=True, description="Tune multi projector if True."
    )
    tune_vl_mixing: bool = Field(default=True, description="Tune vl mixing if True.")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "NitroGenConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(
            num_embodiments, action_dim, hidden_size
        )  # (d -> w)
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )  # (2w -> w)
        self.W3 = CategorySpecificLinear(
            num_embodiments, hidden_size, hidden_size
        )  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding
        # (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat and project
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = F.silu(self.W2(x, cat_ids))

        # 5) Finally W3
        x = self.W3(x, cat_ids)
        return x


class NitroGen(torch.nn.Module):
    config_class = NitroGenConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: NitroGenConfig,
        game_mapping: Dict[str, int] | None = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size

        if "siglip" in config.vision_encoder_name:
            model = SiglipVisionModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder = model.vision_model
            self.vision_encoder_type = "siglip"
        else:
            self.vision_encoder = AutoModel.from_pretrained(config.vision_encoder_name)
            self.vision_encoder_type = "hf_auto"

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        self.model = DiT(config=config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.vl_self_attention_model = SelfAttentionTransformer(
            config=config.vl_self_attention_cfg
        )

        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.hidden_size,
            num_embodiments=config.max_num_embodiments,
        )

        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Placeholder for MM projector if needed, though original had it as None mostly
        self.mm_projector = None

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.game_mapping = game_mapping
        if self.game_mapping is not None:
            self.game_embedding = nn.Embedding(
                len(self.game_mapping),
                self.vision_hidden_size,
                padding_idx=0,
                scale_grad_by_freq=True,
            )

        self.set_trainable_parameters(
            tune_multi_projector=config.tune_multi_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vision_tower=config.tune_vision_tower,
            tune_mm_projector=config.tune_mm_projector,
            tune_vl_mixing=config.tune_vl_mixing,
        )

    def set_trainable_parameters(
        self,
        tune_multi_projector: bool = True,
        tune_diffusion_model: bool = True,
        tune_vision_tower: bool = True,
        tune_mm_projector: bool = True,
        tune_vl_mixing: bool = True,
    ):
        self.tune_multi_projector = tune_multi_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vision_tower = tune_vision_tower
        self.tune_mm_projector = tune_mm_projector
        self.tune_vl_mixing = tune_vl_mixing

        for param in self.parameters():
            param.requires_grad = True

        if self.vision_encoder_type == "siglip":
            # Freeze generic SigLIP layers and head if present.
            try:
                for param in self.vision_encoder.encoder.layers[11].parameters():
                    param.requires_grad = False

                # SiglipVisionModel structure varies; attempt to freeze head if it exists.
                if hasattr(self.vision_encoder, "head"):
                    for param in self.vision_encoder.head.parameters():
                        param.requires_grad = False
            except Exception as e:
                pass  # Squelch if structure differs

        if not tune_multi_projector:
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)

        if not tune_diffusion_model:
            self.model.requires_grad_(False)

        if not tune_vision_tower:
            self.vision_encoder.requires_grad_(False)

        if self.mm_projector is not None and not tune_mm_projector:
            self.mm_projector.requires_grad_(False)

        if not tune_vl_mixing:
            self.vl_self_attention_model.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_multi_projector:
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vision_tower:
                self.vision_encoder.eval()
            if self.mm_projector is not None and not self.tune_mm_projector:
                self.mm_projector.eval()
            if not self.tune_vl_mixing:
                self.vl_self_attention_model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def encode_images(self, images):
        batch_size, num_frames, channels, height, width = images.shape
        images = images.reshape(-1, channels, height, width)

        image_features = self.vision_encoder(images)["last_hidden_state"]
        image_features = rearrange(image_features, "(b f) n d -> b f n d", f=num_frames)

        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)
        return image_features

    def prepare_input_embs(
        self,
        vl_token_ids,
        sa_token_ids,
        vision,
        action,
        dropped_images,
        game_ids=None,
    ):
        B, T = vl_token_ids.shape
        vl_embs = torch.full(
            size=(B, T, self.vision_hidden_size),
            fill_value=0.0,
            dtype=vision.dtype,
            device=vision.device,
        )

        # Vision handling
        tokens_per_image = vision.shape[2]

        # Mask for IMG positions
        # Important: Import constants from tokenizer if needed, or define here.
        # Using hardcoded values matching tokenizer constants for now.
        IMG_TOKEN = 1
        IMG_SEP_TOKEN = 5
        GAME_ID_TOKEN = 6
        ACT_TOKEN = 4

        vision_mask = vl_token_ids == IMG_TOKEN

        # Flatten vision
        vision_flat = vision.reshape(B, -1, self.vision_hidden_size)

        # Select non-dropped
        non_dropped_mask_expanded = (
            (dropped_images == 0)
            .unsqueeze(-1)
            .repeat(1, 1, tokens_per_image)
            .reshape(B, -1)
        )
        valid_vision_embs = vision_flat[non_dropped_mask_expanded]

        # Place
        batch_indices, token_indices = vision_mask.nonzero(as_tuple=True)
        if batch_indices.numel() > 0:
            vl_embs[batch_indices, token_indices] = valid_vision_embs

        # Game ID
        if self.game_mapping is not None and game_ids is not None:
            game_mask = vl_token_ids == GAME_ID_TOKEN
            if game_mask.any():
                game_embs = self.game_embedding(game_ids)
                batch_indices, token_indices = game_mask.nonzero(as_tuple=True)
                vl_embs[batch_indices, token_indices] = game_embs[batch_indices].to(
                    dtype=vl_embs.dtype
                )

        # Action/State
        B_sa, T_sa = sa_token_ids.shape
        sa_embs = torch.full(
            size=(B_sa, T_sa, self.hidden_size),
            fill_value=0.0,
            dtype=action.dtype,
            device=vision.device,
        )

        action_mask = sa_token_ids == ACT_TOKEN
        action_mask_expanded = action_mask.unsqueeze(-1).expand_as(sa_embs)

        # We need to ensure action fits into the mask spaces
        # data["actions"] usually matches the number of ACT tokens exactly if prepared by tokenizer

        sa_embs = sa_embs.masked_scatter(action_mask_expanded, action)

        # Positional Embeddings for SA tokens
        pos_ids = torch.arange(T_sa, dtype=torch.long, device=sa_token_ids.device)
        if self.config.add_pos_embed:
            pos_embs = self.position_embedding(pos_ids)
            pos_embs = pos_embs.unsqueeze(0).expand(B_sa, T_sa, self.hidden_size)
            sa_embs = sa_embs + pos_embs

        return vl_embs, sa_embs

    def forward(self, data: dict) -> dict:
        self.set_frozen_modules_to_eval_mode()

        # Determine device from input data to avoid relying on unstable self.device in DataParallel/PEFT
        device = (
            data["actions"].device
            if "actions" in data
            else (data["images"].device if "images" in data else self.device)
        )

        embodiment_id = data.get("embodiment_id", torch.tensor(0, device=device))

        # Data presence
        has_real_action = data["has_real_action"]

        # 1. Encode Images
        visual_features = self.encode_images(data["images"])

        # 2. Noisy Trajectory
        actions = data["actions"]
        noise = torch.randn_like(actions)
        t = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype
        )
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        # 3. Action Encoder
        action_features = self.action_encoder(
            noisy_trajectory, t_discretized, embodiment_id
        )

        # 4. Prepare Embeddings
        vl_embs, sa_embs = self.prepare_input_embs(
            data["vl_token_ids"],
            data["sa_token_ids"],
            visual_features,
            action_features,
            data["dropped_images"],
            game_ids=data.get("game_ids"),
        )

        vl_embs = self.vl_self_attention_model(vl_embs)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=data.get(
                "vl_attn_mask"
            ),  # DiT handles or ignores None based on internal logic.
            timestep=t_discretized,
        )

        pred = self.action_decoder(model_output, embodiment_id)
        # Prediction corresponds to actions only.
        # The output of DiT is SEQ_LEN of SA tokens.
        # The last N tokens correspond to N action steps.
        # In tokenizer: sa_token_ids are just ACT tokens. So entire sequence is actions.
        pred_actions = pred

        # Loss
        mask = data["actions_mask"]
        raw_loss = F.mse_loss(pred_actions, velocity, reduction="none")
        mask = has_real_action[:, None, None] * mask
        raw_loss = raw_loss * mask
        action_loss = (has_real_action[:, None, None] * raw_loss).sum() / (
            mask.sum() + 1e-6
        )

        return {"loss": action_loss}

    @property
    def device(self):
        try:
            return next(iter(self.parameters())).device
        except StopIteration:
            # Fallback for cases where standard parameter iteration fails (e.g. some PEFT/DP edge cases)
            if hasattr(self, "model") and hasattr(self.model, "parameters"):
                try:
                    return next(iter(self.model.parameters())).device
                except StopIteration:
                    pass
            return torch.device("cpu")

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
