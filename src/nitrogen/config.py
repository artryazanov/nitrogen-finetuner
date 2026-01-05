from typing import Optional

from pydantic import BaseModel, Field

from src.nitrogen.model import NitroGenConfig
from src.nitrogen.tokenizer import NitrogenTokenizerConfig


class ModalityConfig(BaseModel):
    frame_per_sample: int = 1  # number of context frames per sample
    frame_spacing: Optional[int] = (
        None  # frames to skip between each frame. If None, use action_per_chunk
    )
    action_per_chunk: int = 8
    action_shift: int = (
        1  # how many actions to skip between frame[i] and action_chunk[i]
    )
    action_interleaving: bool = (
        False  # if True, action chunks will be interleaved with context frames
    )
    token_set: str = "new"

    def model_post_init(self, __context):
        if self.frame_spacing is None:
            # Set default frame_spacing equal to action_per_chunk if not provided.
            # This handles the field logic manually as Pydantic models are frozen by default in some usages.
            self.frame_spacing = self.action_per_chunk
        assert self.action_shift >= 1, "Frame shift must be at least 1"


class CheckpointConfig(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")

    model_cfg: NitroGenConfig = Field(
        ...,
        description="Model configuration.",
    )
    tokenizer_cfg: NitrogenTokenizerConfig = Field(
        ...,
        description="Tokenizer configuration.",
    )
    modality_cfg: ModalityConfig = Field(
        ...,
        description="Modality configuration for the dataset mixture.",
    )
