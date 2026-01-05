import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from pydantic import BaseModel, Field

# Constants
PAD_TOKEN = 0
IMG_TOKEN = 1
IMG_SEP_TOKEN = 5
LANG_TOKEN = 2
PROPRIO_TOKEN = 3
ACT_TOKEN = 4
GAME_ID_TOKEN = 6
UNCONDITIONAL_ID = None


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, data: dict) -> dict:
        """
        Transform the input data into a tokenized format.
        Args:
            data (dict): Input data containing frames and actions.
        Returns:
            dict: Tokenized data ready for model input.
        """
        pass

    @abstractmethod
    def train(self):
        """Set the tokenizer to training mode."""
        pass

    @abstractmethod
    def eval(self):
        """Set the tokenizer to evaluation mode."""
        pass


class GameMappingConfig(BaseModel):
    src_files: List[str] = Field(
        default_factory=list,
        description="List of source parquet files to build game mapping.",
    )


def get_game_mapping(cfg: GameMappingConfig) -> Dict[str, int]:
    game_set: Set[str] = set()
    for path in cfg.src_files:
        if not os.path.exists(path):
            continue
        try:
            df = pl.read_parquet(path)
            if "game_label" in df.columns:
                for game in df["game_label"].unique():
                    if game == UNCONDITIONAL_ID:
                        continue
                    if game is not None:
                        game_set.add(game)
        except Exception:
            # Squelch errors for now or log warning
            pass

    games = sorted(list(game_set))
    # Set the 0th element to be the unconditional game ID (if we were using None for it, but here we just prepend)
    # Map games to indices starting from 1. Index 0 is reserved for unknown/unconditional.
    mapping = {game: idx + 1 for idx, game in enumerate(games)}
    return mapping


class NitrogenTokenizerConfig(BaseModel):
    tokenizer_id: Literal["nitrogen"] = Field(default="nitrogen", frozen=True)
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    num_visual_tokens_per_frame: int = Field(
        default=256, description="Number of visual tokens per frame."
    )
    max_action_dim: int = Field(default=25, description="Maximum action dimension.")
    max_sequence_length: int = Field(
        default=300, description="Maximum sequence length."
    )
    action_horizon: int = Field(default=16, description="Action horizon.")
    game_mapping_cfg: Optional[GameMappingConfig] = Field(
        default=None, description="Game mapping configuration."
    )
    old_layout: bool = Field(
        default=False,
        description="Whether to use the old layout for actions. If True, [buttons, j_left, j_right]. Else [j_left, j_right, buttons].",
    )


class NitrogenTokenizer(Tokenizer):
    """
    Tokenizer that prepares video and actions into a token-based format.
    """

    def __init__(self, config: NitrogenTokenizerConfig):
        self.training = config.training
        self.num_visual_tokens_per_frame = config.num_visual_tokens_per_frame
        self.max_action_dim = config.max_action_dim
        self.max_sequence_length = config.max_sequence_length
        self.action_horizon = config.action_horizon
        self.old_layout = config.old_layout

        if config.game_mapping_cfg:
            self.game_mapping = get_game_mapping(config.game_mapping_cfg)
        else:
            self.game_mapping = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def _prepare_action(self, data: dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Pad actions to max_action_dim and return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros(
                (self.action_horizon, self.max_action_dim), dtype=bool
            )
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        # Basic validation
        if actions.shape[0] != self.action_horizon:
            # Strictly enforce that the action horizon matches the expected configuration.
            # If this fails, the dataset generation or configuration is likely incorrect.
            assert (
                actions.shape[0] == self.action_horizon
            ), f"Action horizon mismatch: expected {self.action_horizon}, got {actions.shape[0]}"

        n_action_tokens = actions.shape[0]
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(
            actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant"
        )

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def _build_token_ids(
        self, n_images: int, n_action_tokens: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the 1D array of token_ids.
        Returns (vl_token_ids, sa_token_ids).
        """
        vl_token_ids = []
        sa_token_ids = []

        # 0.5) Add a Game ID placeholder
        if self.game_mapping:
            vl_token_ids.append(GAME_ID_TOKEN)

        # 1) Video placeholders
        for _ in range(n_images):
            vl_token_ids.extend([IMG_TOKEN] * self.num_visual_tokens_per_frame)

        # 2) Action tokens
        sa_token_ids.extend([ACT_TOKEN] * n_action_tokens)

        return np.array(vl_token_ids), np.array(sa_token_ids)

    def _prepare_attention_mask(
        self,
        vl_token_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 1D attention mask for vision-language tokens.
        """
        vl_seq_len = vl_token_ids.shape[0]
        vl_attn_mask = np.ones(vl_seq_len, dtype=bool)

        if vl_seq_len > self.max_sequence_length:
            raise ValueError(
                f"VL sequence length {vl_seq_len} exceeds max {self.max_sequence_length}!"
            )

        left_pad_len = self.max_sequence_length - vl_seq_len

        # Pad token_ids (with PAD_TOKEN)
        vl_token_ids = np.pad(
            vl_token_ids, (left_pad_len, 0), constant_values=PAD_TOKEN
        )

        # Pad attention mask with 0 (padding tokens)
        vl_attn_mask = np.pad(vl_attn_mask, (left_pad_len, 0), constant_values=0)

        return vl_token_ids, vl_attn_mask

    def pack_actions(
        self, buttons: np.ndarray, j_left: np.ndarray, j_right: np.ndarray
    ) -> np.ndarray:
        """
        Pack discrete buttons and continuous joysticks into a single action tensor.
        """
        # Normalize joysticks to [0, 1]
        j_left = (j_left + 1) / 2.0
        j_right = (j_right + 1) / 2.0

        # Concatenate: [buttons, j_left, j_right] or [j_left, j_right, buttons]
        # logic from original 'pack_actions':
        # buttons, j_left, j_right concatenated axis=-1
        # Original code: np.concatenate([buttons,j_left,j_right],axis=-1, ...)

        # Pack using the standard [buttons, j_left, j_right] layout.
        # This implementation assumes the new layout is used.

        action = np.concatenate([buttons, j_left, j_right], axis=-1, dtype=np.float32)

        # Squeeze the first dimension if it exists and is 1 (original code comment: "number of chunks, which is 1 here")
        if action.ndim == 3 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    def encode(self, data: dict) -> dict:
        """
        Main entry point for the transform.
        """
        transformed_data = {**data}  # Start with a copy

        # n_images is count of non-dropped frames
        n_images = np.sum(data["dropped_frames"] == False)
        transformed_data["images"] = data["frames"]
        transformed_data["dropped_images"] = data["dropped_frames"]

        if self.training:
            # We construct 'action' from raw components if provided
            # This is typically what limits this to training time or when we have ground truth
            if "buttons" in data:
                packed_actions = self.pack_actions(
                    data["buttons"], data["j_left"], data["j_right"]
                )
                data["action"] = packed_actions

            transformed_data["has_real_action"] = np.ones((), dtype=bool)

            actions, actions_mask, n_action_tokens = self._prepare_action(data)
            transformed_data["actions"] = actions
            transformed_data["actions_mask"] = actions_mask

            # Validation
            if (
                transformed_data["actions"].shape
                != transformed_data["actions_mask"].shape
            ):
                raise ValueError("Shape mismatch between actions and mask")
        else:
            n_action_tokens = self.action_horizon

        transformed_data["has_detection_target"] = np.zeros((), dtype=bool)

        # Build token_ids
        vl_token_ids, sa_token_ids = self._build_token_ids(n_images, n_action_tokens)

        # Build attention maks
        vl_token_ids, vl_attn_mask = self._prepare_attention_mask(vl_token_ids)

        transformed_data["vl_token_ids"] = vl_token_ids
        transformed_data["sa_token_ids"] = sa_token_ids
        transformed_data["vl_attn_mask"] = vl_attn_mask
        transformed_data["embodiment_id"] = torch.tensor(0, dtype=torch.long)

        if self.game_mapping:
            game_name = data.get("game")
            if game_name in self.game_mapping:
                transformed_data["game_ids"] = torch.tensor(
                    self.game_mapping[game_name], dtype=torch.long
                )
            else:
                # Default to 0 if not found (unknown/unconditional)
                transformed_data["game_ids"] = torch.tensor(0, dtype=torch.long)
        else:
            transformed_data["game_ids"] = torch.tensor(0, dtype=torch.long)

        return transformed_data
