from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="nvidia/NitroGen",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading the model."},
    )
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA (Low-Rank Adaptation)."}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="artryazanov/nitrogen-bizhawk-nes-felix-the-cat-world-1",
        metadata={"help": "The name of the dataset to use."},
    )
    prediction_horizon: int = field(
        default=16,
        metadata={
            "help": "Number of future frames to predict actions for (matches model output shape)."
        },
    )
    analog_deadzone: float = field(
        default=0.05,
        metadata={
            "help": "Threshold below which analog stick values are treated as 0.0."
        },
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(default="models/checkpoints")
    learning_rate: float = field(default=1e-4)  # Slightly lower for DiT fine-tuning
    per_device_train_batch_size: int = field(default=8)
    num_train_epochs: int = field(default=3)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    fp16: bool = field(default=True)


# Schema Definition for the 21-dimensional vector
# Indices:
# 0: Left Stick X
# 1: Left Stick Y
# 2: Right Stick X
# 3: Right Stick Y
# 4-20: Buttons
class ActionSchema:
    L_STICK_X = 0
    L_STICK_Y = 1
    R_STICK_X = 2
    R_STICK_Y = 3

    # 17 Buttons
    BUTTON_ORDER = [
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

    BUTTON_START_IDX = 4
