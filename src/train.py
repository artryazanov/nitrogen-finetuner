import os
import sys
from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel  # Use AutoModel for custom architectures
from transformers import (
    AutoProcessor,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import (
    set_seed,
)
from huggingface_hub import snapshot_download

from src.config import ActionSchema, DataArguments, ModelArguments, TrainingArguments
from src.data import SlidingWindowDataset, UniversalVectorProcessor


class DiTTrainer(Trainer):
    """
    Custom Trainer to handle the regression loss for the NitroGen DiT.
    We compute MSE between predicted actions and ground truth actions.
    """

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # inputs contains 'pixel_values' and 'actions' (from our SlidingWindowDataset)
        labels = inputs.pop("actions")  # Shape: (B, 16, 21)

        # Forward pass
        # We assume the model's forward signature accepts pixel_values and optionally 'labels' or 'actions'
        # If the model is a standard HF implementation of DiT, it might return a specific output class.
        # We'll treat it generically.
        outputs = model(**inputs)

        # If the model computes loss internally, use it.
        # This is common in custom HF models.
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
            logits = getattr(outputs, "logits", None)
        else:
            # If the model returns raw predictions (e.g. logits/pred_actions), compute MSE.
            # We assume output shape is matching (B, 16, 21) or compatible
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Regression Loss: MSE
            # Ensure shapes match
            if logits.shape != labels.shape:
                # This might happen if the model expects different shaping.
                # For this implementation, we assume strict compatibility.
                # In a real debug scenario, we'd log shapes here.
                pass

            loss = F.mse_loss(logits, labels)

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

    # 2. Load Processor and Model
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Determine local path for the model
    model_name_slug = model_args.model_name_or_path.replace("/", "--")
    local_model_path = os.path.join("models", model_name_slug)

    if not os.path.exists(local_model_path):
        logger.info(f"Model not found locally at {local_model_path}. Downloading...")
        try:
            snapshot_download(
                repo_id=model_args.model_name_or_path,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise e
    else:
        logger.info(f"Model found locally at {local_model_path}.")

    # Load separate components if needed, or use AutoProcessor if the hub repo is fully integrated
    try:
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            trust_remote_code=model_args.trust_remote_code,
        )
    except Exception as e:
        logger.warning(
            f"Could not load AutoProcessor: {e}. Fallback logic might be needed for image transforms."
        )
        # Minimal Fallback (assuming standard SigLip/ViT transforms)
        from transformers import SiglipImageProcessor

        processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )

    # Load Model
    torch_dtype = torch.float16 if train_args_dataclass.fp16 else torch.float32

    # We use AutoModel because NitroGen likely doesn't fit a standard task head class like CausalLM perfectly
    model = AutoModel.from_pretrained(
        local_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto",
    )

    # 3. Apply LoRA
    if model_args.use_lora:
        logger.info("Applying LoRA configuration...")
        # Target modules: We try to target linear layers in the DiT or ViT.
        # Standard names: q, k, v, proj, fc, dense
        peft_config = LoraConfig(
            task_type=None,  # Custom task
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
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 4. Prepare Dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    raw_dataset = transformers.datasets.load_dataset(
        data_args.dataset_name, split="train"
    )

    # Initialize our Universal Processor
    vector_processor = UniversalVectorProcessor(deadzone=data_args.analog_deadzone)

    # Create Sliding Window Wrapper
    train_window_dataset = SlidingWindowDataset(
        hf_dataset=raw_dataset,
        image_processor=processor,  # Takes RGB PIL -> Tensor
        processor=vector_processor,
        horizon=data_args.prediction_horizon,
    )

    # Split manually if needed, or just use slice
    # For this demo, we'll just train on the whole thing or a subset
    # eval_dataset = ...

    # 5. Define Trainer
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
        report_to="wandb",
        run_name="nitrogen-finetuner-dit",
        # Custom collator might be needed to stack tensors
        # Default usually handles dict of tensors well
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
