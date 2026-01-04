from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from src.config import ActionSchema


class UniversalVectorProcessor:
    """
    Transforms raw dataset rows (booleans/lists) into a (21,) float tensor
    adhering to the NitroGen DiT spec.
    """

    def __init__(self, deadzone: float = 0.05):
        self.deadzone = deadzone

    def _apply_deadzone(self, val: float) -> float:
        return 0.0 if abs(val) < self.deadzone else val

    def get_vector(self, row: Dict[str, Any]) -> torch.Tensor:
        """
        Constructs the (21,) action vector.
        """
        vector = torch.zeros(21, dtype=torch.float32)

        # 1. Analog Sticks (Dimensions 0-3)
        # Left Stick (j_left)
        j_left = row.get("j_left", [0.0, 0.0])
        if j_left and len(j_left) >= 2:
            vector[ActionSchema.L_STICK_X] = self._apply_deadzone(j_left[0])
            vector[ActionSchema.L_STICK_Y] = self._apply_deadzone(j_left[1])

        # Right Stick (j_right)
        j_right = row.get("j_right", [0.0, 0.0])
        if j_right and len(j_right) >= 2:
            vector[ActionSchema.R_STICK_X] = self._apply_deadzone(j_right[0])
            vector[ActionSchema.R_STICK_Y] = self._apply_deadzone(j_right[1])

        # 2. Buttons (Dimensions 4-20)
        for i, button_name in enumerate(ActionSchema.BUTTON_ORDER):
            idx = ActionSchema.BUTTON_START_IDX + i
            # Get bool value, convert to float (1.0 or 0.0)
            is_pressed = row.get(button_name, False)
            vector[idx] = 1.0 if is_pressed else 0.0

        return vector


class SlidingWindowDataset(Dataset):
    """
    Wraps a Hugging Face dataset to provide a sliding window of actions.
    Input: Image at index t
    Target: Actions at indices [t ... t + horizon]
    """

    def __init__(
        self,
        hf_dataset,
        image_processor,
        processor: UniversalVectorProcessor,
        horizon: int = 16,
    ):
        self.dataset = hf_dataset
        self.image_processor = image_processor
        self.processor = processor
        self.horizon = horizon

        # We can't use the last (horizon - 1) frames as start points
        self.valid_length = len(hf_dataset) - horizon + 1

    def __len__(self):
        return max(0, self.valid_length)

    def __getitem__(self, idx):
        # 1. Get the Image at the current timestep
        # We access the raw dataset row
        current_row = self.dataset[idx]
        image = current_row["image"].convert("RGB")

        # Process image using the model's image processor (SigLip)
        if self.image_processor:
            pixel_values = self.image_processor(
                image, return_tensors="pt"
            ).pixel_values.squeeze(0)
        else:
            # Fallback (should not happen if generic processor provided)
            raise ValueError("Image processor required for SigLip encoding")

        # 2. Get the Action Sequence (Horizon)
        # We need to slice the dataset. HF Datasets support slicing efficiently.
        # slice_rows is a dict of lists: {'south': [val, val...], 'image': [...]}
        window_rows_dict = self.dataset[idx : idx + self.horizon]

        # We need to convert this "columnar" slice back to "row" format for our processor
        # Or optimize the processor to handle batches.
        # For clarity/safety, we reconstruct rows.
        actions_list = []
        for i in range(self.horizon):
            # Reconstruct single row dict for the processor
            row_snapshot = {
                k: window_rows_dict[k][i] for k in window_rows_dict if k != "image"
            }
            action_vec = self.processor.get_vector(row_snapshot)
            actions_list.append(action_vec)

        # Stack into (16, 21) tensor
        actions_tensor = torch.stack(actions_list)

        return {
            "pixel_values": pixel_values,
            "actions": actions_tensor,  # Model expects 'actions' key usually, or 'labels' depending on implementation
        }


if __name__ == "__main__":
    # verification mock
    print("Testing UniversalVectorProcessor...")
    proc = UniversalVectorProcessor()
    mock_row = {"south": True, "j_left": [0.55, -0.01]}  # -0.01 should be deadzoned
    vec = proc.get_vector(mock_row)
    print(f"Shape: {vec.shape}")  # Should be (21,)
    print(f"South (idx 4): {vec[4]}")  # Should be 1.0
    print(f"L_Stick X (idx 0): {vec[0]}")  # Should be 0.55
    print(f"L_Stick Y (idx 1): {vec[1]}")  # Should be 0.0 (deadzone)
    assert vec.shape == (21,)
    print("Test Passed!")
