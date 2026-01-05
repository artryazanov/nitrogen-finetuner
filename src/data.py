from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.config import ActionSchema

# Note: UniversalVectorProcessor is replaced by NitrogenTokenizer logic


class SlidingWindowDataset(Dataset):
    """
    Wraps a Hugging Face dataset to provide a sliding window of actions.
    Input: Image at index t (and optionally history if configured)
    Target: Actions at indices [t ... t + horizon]
    """

    def __init__(
        self,
        hf_dataset,
        image_processor,
        tokenizer,
        horizon: int = 16,
    ):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.horizon = horizon

        # We can't use the last (horizon - 1) frames as start points
        self.valid_length = len(hf_dataset) - horizon + 1

    def __len__(self):
        return max(0, self.valid_length)

    def __getitem__(self, idx):
        # 1. Get the Image at the current timestep
        current_row = self.dataset[idx]
        image = current_row["image"].convert("RGB")

        # Process image using the provided image processor
        # Returns [1, C, H, W] if typical processor, or just [C, H, W]
        # We need [T, C, H, W] for the tokenizer/model where T=1
        if self.image_processor:
            # processor returns Batch x Channel x Height x Width usually if list passed
            # Here we pass single image
            pixel_values = self.image_processor(
                image, return_tensors="pt"
            ).pixel_values  # [1, 3, H, W]
        else:
            raise ValueError("Image processor required")

        # 2. Get the Action Sequence (Horizon)
        window_rows_dict = self.dataset[idx : idx + self.horizon]

        # Extract features for tokenizer
        # Joysticks: (Horizon, 2)
        # Handle cases where column might be missing or None
        def get_col(name, default_val):
            val = window_rows_dict.get(name)
            if val is None:
                return [default_val] * self.horizon
            return [v if v is not None else default_val for v in val]

        j_left = np.array(get_col("j_left", [0.0, 0.0]), dtype=np.float32)
        j_right = np.array(get_col("j_right", [0.0, 0.0]), dtype=np.float32)

        # Ensure shape (Horizon, 2)
        if j_left.ndim == 1:
            # If data is weirdly list of list
            pass
        # Basic check
        if j_left.shape != (self.horizon, 2):
            # Try to fix?
            # Assuming dataset is correct for now
            pass

        # Buttons: (Horizon, NumButtons)
        buttons_list = []
        for button_name in ActionSchema.BUTTON_ORDER:
            # list of bools
            vals = get_col(button_name, False)
            # Convert to float 0.0/1.0
            float_vals = [1.0 if v else 0.0 for v in vals]
            buttons_list.append(float_vals)

        buttons = np.array(buttons_list, dtype=np.float32).T  # [Horizon, NumButtons]

        # Prepare Data Dict for NitrogenTokenizer
        # It expects raw numpy/torch inputs
        data = {
            # frames: [T, C, H, W] - we have [1, C, H, W] from processor
            "frames": pixel_values,
            "dropped_frames": np.array([False]),  # 1 frame, present
            "game": "unconditional",  # Default
            # Inputs: (Horizon, Dims) -> Tokenizer expects (1, Horizon, Dims)
            "buttons": np.expand_dims(buttons, 0),
            "j_left": np.expand_dims(j_left, 0),
            "j_right": np.expand_dims(j_right, 0),
        }

        # Tokenize (returns dict of tensors)
        tokenized = self.tokenizer.encode(data)

        # Ensure inputs are appropriate tensors (tokenizer usually returns tensors or numpy)
        # We want to return a dict that the collator can stack.
        # NitrogenTokenizer returns dict with keys like 'vl_token_ids', 'sa_token_ids', 'actions', etc.
        # We also need to pass 'has_real_action' etc if tokenizer set them.

        # Convert any numpy to torch if not already
        final_dict = {}
        for k, v in tokenized.items():
            if isinstance(v, np.ndarray):
                final_dict[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                final_dict[k] = v
            else:
                final_dict[k] = v

        return final_dict
