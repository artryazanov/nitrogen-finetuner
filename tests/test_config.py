import pytest

from src.config import ActionSchema, DataArguments, ModelArguments, TrainingArguments


def test_action_schema_constants():
    """Test that ActionSchema constants are defined correctly."""
    assert ActionSchema.L_STICK_X == 0
    assert ActionSchema.L_STICK_Y == 1
    assert ActionSchema.R_STICK_X == 2
    assert ActionSchema.R_STICK_Y == 3
    assert ActionSchema.BUTTON_START_IDX == 4

    expected_buttons = [
        "south",
        "east",
        "west",
        "north",
        "left_shoulder",
        "right_shoulder",
        "left_trigger",
        "right_trigger",
        "start",
        "back",
        "guide",
        "dpad_up",
        "dpad_down",
        "dpad_left",
        "dpad_right",
        "left_thumb",
        "right_thumb",
    ]
    assert ActionSchema.BUTTON_ORDER == expected_buttons
    assert len(ActionSchema.BUTTON_ORDER) == 17


def test_model_arguments_defaults(model_args):
    """Test default values for ModelArguments."""
    assert model_args.model_name_or_path == "nvidia/NitroGen"
    assert model_args.trust_remote_code is True
    assert model_args.use_lora is True


def test_data_arguments_defaults(data_args):
    """Test default values for DataArguments."""
    assert (
        data_args.dataset_name
        == "artryazanov/nitrogen-bizhawk-nes-felix-the-cat-world-1"
    )
    assert data_args.prediction_horizon == 16
    assert data_args.analog_deadzone == 0.05


def test_training_arguments_defaults(training_args):
    """Test default values for TrainingArguments."""
    assert training_args.output_dir == "models/checkpoints"
    assert training_args.learning_rate == 1e-4
    assert training_args.per_device_train_batch_size == 8
    assert training_args.num_train_epochs == 3
    assert training_args.fp16 is True
