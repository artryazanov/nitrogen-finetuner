import os
import sys
from typing import Any, Dict, Union

import datasets
import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import (
    set_seed,
)

from src.config import ActionSchema, DataArguments, ModelArguments, TrainingArguments
from src.data import SlidingWindowDataset
from src.nitrogen.config import CheckpointConfig

# Native NitroGen imports
from src.nitrogen.model import NitroGen
from src.nitrogen.tokenizer import NitrogenTokenizer


class DiTTrainer(Trainer):
    """
    Custom Trainer for NitroGen DiT.
    The model computes loss internally in its forward pass.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # NitroGen.forward expects 'data' dict which matches our inputs
        # It returns {'loss': ...}
        outputs = model(inputs)

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def main():
    # 1. Parse Arguments
    # In a full CLI app we'd use HfArgumentParser, here we manually instantiate for clarity/defaults
    model_args = ModelArguments()
    data_args = DataArguments()
    train_args_dataclass = TrainingArguments()

    # Setup Logging
    transformers.utils.logging.set_verbosity_info()
    logger = transformers.utils.logging.get_logger("transformers")

    set_seed(42)

    # 2. Load Model using Native Class
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_name_slug = model_args.model_name_or_path.replace("/", "--")
    local_model_path = os.path.join("models", model_name_slug)

    # Download if not exists
    if not os.path.exists(local_model_path):
        logger.info(f"Model not found locally at {local_model_path}. Downloading...")
        try:
            snapshot_download(
                repo_id=model_args.model_name_or_path,
                local_dir=local_model_path,
            )
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise e
    else:
        logger.info(f"Model found locally at {local_model_path}.")

    # Load Checkpoint using Native Logic
    ckpt_path = os.path.join(local_model_path, "ng.pt")
    if not os.path.exists(ckpt_path):
        # Fallback to pytorch_model.bin if renamed
        fallback_path = os.path.join(local_model_path, "pytorch_model.bin")
        if os.path.exists(fallback_path):
            logger.info("ng.pt not found, falling back to pytorch_model.bin")
            ckpt_path = fallback_path

    logger.info(f"Loading custom NitroGen checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # Extract config
    ckpt_config = CheckpointConfig.model_validate(checkpoint["ckpt_config"])
    model_cfg = ckpt_config.model_cfg
    tokenizer_cfg = ckpt_config.tokenizer_cfg

    logger.info(f"Model Config Loaded: {model_cfg.model_type}")

    # Init model
    model = NitroGen(config=model_cfg, game_mapping=None)

    # Patch config for Peft compatibility using a wrapper (Pydantic models are strict)
    class ConfigWrapper:
        def __init__(self, config):
            self._config = config

        def __getattr__(self, name):
            return getattr(self._config, name)

        def get(self, key, default=None):
            return getattr(self._config, key, default)

        def to_dict(self):
            return self._config.model_dump()

    model.config = ConfigWrapper(model.config)

    # Load weights
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(
        f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )

    # Move to float32 or bf16 depending on training args, usually float32 for training stability then AMP
    model.to(dtype=torch.float32)

    # 3. Apply LoRA
    if model_args.use_lora:
        logger.info("Applying LoRA configuration...")
        peft_config = LoraConfig(
            task_type=None,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "proj_out",
                "dense",
            ],
            modules_to_save=[],  # Typically we dont save head for LoRA unless needed, but here head is part of NitroGen architecture
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 4. Prepare Dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")

    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    dataset_name_slug = data_args.dataset_name.replace("/", "--")
    local_dataset_path = os.path.join("datasets", dataset_name_slug)

    try:
        if os.path.exists(local_dataset_path):
            logger.info(
                f"Dataset found locally at {local_dataset_path}. Loading from disk..."
            )
            raw_dataset = datasets.load_from_disk(local_dataset_path)
        else:
            logger.info(f"Dataset not found locally. Downloading from Hub...")
            raw_dataset = datasets.load_dataset(data_args.dataset_name, split="train")
            logger.info(f"Saving dataset to {local_dataset_path}...")
            raw_dataset.save_to_disk(local_dataset_path)

    except Exception as e:
        logger.warning(f"Failed to load dataset normally: {e}")
        raise e

    # Init Image Processor
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_cfg.vision_encoder_name
        )
    except Exception:
        # Fallback
        image_processor = AutoImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    # Init Tokenizer
    tokenizer = NitrogenTokenizer(tokenizer_cfg)

    # Create Sliding Window Wrapper
    train_window_dataset = SlidingWindowDataset(
        hf_dataset=raw_dataset,
        image_processor=image_processor,
        tokenizer=tokenizer,
        horizon=tokenizer.action_horizon,
    )

    # 5. Define Trainer
    # Check for WandB
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "none":
        logger.info("WANDB_API_KEY not found. Disabling WandB logging.")

    hf_train_args = HFTrainingArguments(
        output_dir=train_args_dataclass.output_dir,
        num_train_epochs=train_args_dataclass.num_train_epochs,
        per_device_train_batch_size=train_args_dataclass.per_device_train_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=train_args_dataclass.learning_rate,
        logging_steps=train_args_dataclass.logging_steps,
        save_steps=train_args_dataclass.save_steps,
        fp16=train_args_dataclass.fp16,
        remove_unused_columns=False,  # Critical for custom Datasets returning unknown keys like 'actions'
        report_to=report_to,
        run_name="nitrogen-finetuner-dit" if report_to != "none" else None,
    )

    trainer = DiTTrainer(
        model=model,
        args=hf_train_args,
        train_dataset=train_window_dataset,
    )

    # 6. Train
    logger.info("Starting training...")
    trainer.train()

    # 7. Save
    logger.info("Saving...")
    trainer.save_model(os.path.join(train_args_dataclass.output_dir, "final_model"))


if __name__ == "__main__":
    main()
