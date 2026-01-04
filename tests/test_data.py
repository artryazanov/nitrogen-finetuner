from unittest.mock import MagicMock

import pytest
import torch

from src.config import ActionSchema
from src.data import SlidingWindowDataset, UniversalVectorProcessor

# --- UniversalVectorProcessor Tests ---


def test_processor_deadzone():
    """Test deadzone application in UniversalVectorProcessor."""
    processor = UniversalVectorProcessor(deadzone=0.1)

    # Value inside deadzone should be 0.0
    assert processor._apply_deadzone(0.05) == 0.0
    assert processor._apply_deadzone(-0.05) == 0.0

    # Value outside deadzone should remain
    assert processor._apply_deadzone(0.15) == 0.15
    assert processor._apply_deadzone(-0.15) == -0.15


def test_processor_get_vector_sticks():
    """Test stick vectorization."""
    processor = UniversalVectorProcessor(deadzone=0.0)

    row = {"j_left": [0.5, -0.5], "j_right": [0.1, 0.9]}
    vector = processor.get_vector(row)

    assert vector[ActionSchema.L_STICK_X] == 0.5
    assert vector[ActionSchema.L_STICK_Y] == -0.5
    assert vector[ActionSchema.R_STICK_X] == 0.1
    assert vector[ActionSchema.R_STICK_Y] == 0.9


def test_processor_get_vector_buttons():
    """Test button vectorization."""
    processor = UniversalVectorProcessor()

    # Buttons from config.py: south, east, etc.
    # Let's press 'south' and 'start'
    row = {"south": True, "start": True, "east": False}
    vector = processor.get_vector(row)

    # Calculate expected indices
    south_idx = ActionSchema.BUTTON_START_IDX + ActionSchema.BUTTON_ORDER.index("south")
    start_idx = ActionSchema.BUTTON_START_IDX + ActionSchema.BUTTON_ORDER.index("start")

    assert vector[south_idx] == 1.0
    assert vector[start_idx] == 1.0

    # Verify a non-pressed button
    east_idx = ActionSchema.BUTTON_START_IDX + ActionSchema.BUTTON_ORDER.index("east")
    assert vector[east_idx] == 0.0


def test_processor_missing_keys():
    """Test behavior when keys are missing from the row."""
    processor = UniversalVectorProcessor()
    row = {}
    vector = processor.get_vector(row)

    assert vector.shape == (21,)
    assert torch.all(vector == 0.0)


# --- SlidingWindowDataset Tests ---


@pytest.fixture
def mock_image_processor():
    processor = MagicMock()
    # Mock return value of processor(image, return_tensors="pt")
    mock_output = MagicMock()
    mock_output.pixel_values = torch.randn(1, 3, 224, 224)
    processor.return_value = mock_output
    return processor


@pytest.fixture
def mock_dataset():
    # Create a dummy list-based dataset
    # We need enough items for a horizon
    horizon = 4
    num_samples = 10

    data = []
    for i in range(num_samples):
        item = {
            "image": MagicMock(),  # Mock PIL image
            "south": True if i % 2 == 0 else False,  # Alternate button press
            "j_left": [float(i) / num_samples, 0.0],
        }
        # Add other required button keys with False to avoid KeyErrors if the code expects them?
        # The code uses row.get(button_name, False), so missing keys are fine.
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
                keys = sliced_data[0].keys()
                result = {k: [d[k] for d in sliced_data] for k in keys}
                return result
            else:
                return self.data[idx]

    return MockHFDataset(data)


def test_sliding_window_len(mock_dataset, mock_image_processor):
    """Test dataset length calculation."""
    horizon = 4
    dataset = SlidingWindowDataset(
        hf_dataset=mock_dataset,
        image_processor=mock_image_processor,
        processor=UniversalVectorProcessor(),
        horizon=horizon,
    )

    # Expected len = total - horizon + 1
    # 10 - 4 + 1 = 7
    assert len(dataset) == 7


def test_sliding_window_getitem(mock_dataset, mock_image_processor):
    """Test getting an item returns correct shapes."""
    horizon = 4
    dataset = SlidingWindowDataset(
        hf_dataset=mock_dataset,
        image_processor=mock_image_processor,
        processor=UniversalVectorProcessor(),
        horizon=horizon,
    )

    item = dataset[0]

    # Check keys
    assert "pixel_values" in item
    assert "actions" in item

    # Check shapes
    # pixel_values comes from mock_image_processor: (1, 3, 224, 224) -> squeezed to (3, 224, 224)
    assert item["pixel_values"].shape == (3, 224, 224)

    # actions: (horizon, 21)
    assert item["actions"].shape == (horizon, 21)
