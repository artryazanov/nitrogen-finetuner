from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.config import ActionSchema
from src.data import SlidingWindowDataset


@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    # Mock return value of processor(image, return_tensors="pt")
    mock_output = MagicMock()
    # Batch size 1, channels 3, 224x224
    mock_output.pixel_values = torch.randn(1, 3, 224, 224)
    processor.return_value = mock_output
    return processor


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # Mock encode return value
    # NitrogenTokenizer.encode returns a dict of tensors/arrays
    tokenizer.encode.return_value = {
        "visual_tokens": torch.randint(0, 1000, (1, 256)),
        "action_tokens": torch.randint(0, 100, (1, 16)),
        "actions": torch.randn(1, 16, 21),  # (Batch, Horizon, Dim)
    }
    return tokenizer


@pytest.fixture
def mock_dataset():
    # Create a dummy list-based dataset
    # We need enough items for a horizon
    num_samples = 10

    data = []
    for i in range(num_samples):
        item = {
            "image": MagicMock(),  # Mock PIL image
            "south": True if i % 2 == 0 else False,  # Alternate button press
            "j_left": [float(i) / num_samples, 0.0],
            # Add j_right and other buttons if necessary for dataset logic,
            # or rely on get_col default values in SlidingWindowDataset
        }
        data.append(item)

    # The dataset needs to support slicing [start:end] -> returns dict of lists
    class MockHFDataset:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                # Return dict of lists
                sliced_data = self.data[idx]
                if not sliced_data:
                    return {}
                keys = sliced_data[0].keys()
                result = {k: [d[k] for d in sliced_data] for k in keys}
                return result
            else:
                return self.data[idx]

    return MockHFDataset(data)


def test_sliding_window_len(mock_dataset, mock_image_processor, mock_tokenizer):
    """Test dataset length calculation."""
    horizon = 4
    dataset = SlidingWindowDataset(
        hf_dataset=mock_dataset,
        image_processor=mock_image_processor,
        tokenizer=mock_tokenizer,
        horizon=horizon,
    )

    # Expected len = total - horizon + 1
    # 10 - 4 + 1 = 7
    assert len(dataset) == 7


def test_sliding_window_getitem(mock_dataset, mock_image_processor, mock_tokenizer):
    """Test getting an item returns correct shapes."""
    horizon = 4
    dataset = SlidingWindowDataset(
        hf_dataset=mock_dataset,
        image_processor=mock_image_processor,
        tokenizer=mock_tokenizer,
        horizon=horizon,
    )

    item = dataset[0]

    # Verify tokenizer.encode was called with correct structure
    mock_tokenizer.encode.assert_called_once()
    call_args = mock_tokenizer.encode.call_args[0][0]  # First arg is 'data' dict

    # Check data dict keys passed to tokenizer
    assert "frames" in call_args
    assert "buttons" in call_args
    assert "j_left" in call_args
    assert "j_right" in call_args

    # Check actions shape passed to tokenizer
    # buttons should be (1, Horizon, NumButtons)
    assert call_args["buttons"].shape == (1, horizon, 17)  # 17 buttons in ActionSchema

    # Check item returned by dataset (should be whatever tokenizer returned)
    assert "visual_tokens" in item
    assert "action_tokens" in item
