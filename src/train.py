import os
import sys
from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
import datasets
from transformers import AutoModel, AutoConfig  # Use AutoModel for custom architectures
from transformers import (
    AutoProcessor,
    Trainer,
)
from transformers import TrainingArguments as HFTrainingArguments
from transformers import (
    set_seed,
)

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
        # DiaModel requires 'input_ids' or 'text_encodings'. 
        # Since we are fine-tuning on visual-action tasks, we provide a dummy text prompt (BOS token).
        if "input_ids" not in inputs:
            batch_size = inputs["pixel_values"].shape[0]
            # Use model.config.bos_token_id if available, otherwise default to 0
            # Handle DataParallel wrapping where model might be wrapped
            # unwrapped_model = getattr(model, "module", model)
            # encoder_vocab = getattr(unwrapped_model.config.encoder_config, "vocab_size", 256)
            
            # The NitroGen Encoder has a small vocab (256), but bos_token_id is 1026.
            # Passing bos_token_id causes CUDA index out of bounds.
            # We use 0 as a safe dummy token.
            safe_id = 0
            
            dummy_ids = torch.full(
                (batch_size, 1), 
                safe_id, 
                dtype=torch.long, 
                device=inputs["pixel_values"].device
            )
            inputs["input_ids"] = dummy_ids

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
            # outputs[0] is usually the last hidden state: [Batch, SeqLen, Hidden] -> [B, 1, 1024]
            hidden_states = outputs[0]
            
            # Use our custom action_head to project hidden -> actions
            # Need to access the head from the base model (unwrapped from DDP/Peft)
            # DDP wraps in .module
            # PeftModel wraps base in .base_model or directly forwards attributes if possible (but DDP blocks it)
            
            # Unwrap DDP
            base_model = getattr(model, "module", model)
            # Unwrap Peft if needed, or rely on attribute access. 
            # PeftModel usually has 'base_model', which has 'model' (AutoModel).
            # But we attached 'action_head' to the AutoModel instance 'model'.
            # So if PeftModel is used: model.action_head might work if Peft forwards it, 
            # or we need model.base_model.model.action_head.
            
            # Robust way to find action_head
            action_head = getattr(base_model, "action_head", None)
            if action_head is None:
                # Try digging into Peft
                if hasattr(base_model, "base_model"):
                    action_head = getattr(base_model.base_model, "action_head", None)
                    # If still None, maybe one more level
                    if action_head is None and hasattr(base_model.base_model, "model"):
                         action_head = getattr(base_model.base_model.model, "action_head", None)
            
            if action_head is None:
                raise AttributeError("Could not find 'action_head' in model. Ensure it was added in main().")

            # Project: [B, 1, 1024] -> [B, 1, 336]
            # Ensure input is cast to head's dtype (float32) to avoid mixed precision issues in this custom layer
            hidden_states = hidden_states.to(action_head.weight.dtype)
            logits = action_head(hidden_states)
            
            # Reshape to [Batch, Horizon, ActionDim] -> [B, 16, 21]
            B, H, A = labels.shape
            logits = logits.view(B, H, A)

            # Regression Loss: MSE
            # Cast to float32 for stability and to avoid mixed precision issues in backward
            loss = F.mse_loss(logits.float(), labels.float())

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
            )
            # Rename ng.pt to pytorch_model.bin if it exists and the bin file doesn't
            ng_pt_path = os.path.join(local_model_path, "ng.pt")
            bin_path = os.path.join(local_model_path, "pytorch_model.bin")
            if os.path.exists(ng_pt_path) and not os.path.exists(bin_path):
                logger.info("Renaming ng.pt to pytorch_model.bin for AutoModel compatibility.")
                os.rename(ng_pt_path, bin_path)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise e
    else:
        logger.info(f"Model found locally at {local_model_path}.")
        
    # Ensure compatibility by renaming ng.pt if needed (even if locally found)
    ng_pt_path = os.path.join(local_model_path, "ng.pt")
    bin_path = os.path.join(local_model_path, "pytorch_model.bin")
    if os.path.exists(ng_pt_path) and not os.path.exists(bin_path):
        logger.info("Renaming ng.pt to pytorch_model.bin for AutoModel compatibility.")
        os.rename(ng_pt_path, bin_path)

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

    # Fix weights for DiaModel compatibility
    state_dict = None
    if os.path.exists(bin_path):
        logger.info("Loading and fixing state_dict for DiaModel...")
        try:
            raw_sd = torch.load(bin_path, map_location="cpu")
            if "state_dict" in raw_sd:
                raw_sd = raw_sd["state_dict"]
            elif "model" in raw_sd:
                raw_sd = raw_sd["model"]
            
            new_sd = {}
            for k, v in raw_sd.items():
                new_k = k
                # Strip common prefixes
                if new_k.startswith("model."):
                    new_k = new_k[6:]
                elif new_k.startswith("module."):
                    new_k = new_k[7:]

                # Mapping Logic for NitroGen (ng.pt) -> DiaModel
                # Encoder / Decoder mapping
                # We map 'vl_self_attention_model' (backbone) to 'decoder' (Dia Decoder)
                if "vl_self_attention_model.transformer_blocks" in new_k:
                    new_k = new_k.replace("vl_self_attention_model.transformer_blocks", "decoder.layers")
                    
                    # Attention
                    new_k = new_k.replace("attn1.to_q", "self_attention.q_proj")
                    new_k = new_k.replace("attn1.to_k", "self_attention.k_proj")
                    new_k = new_k.replace("attn1.to_v", "self_attention.v_proj")
                    new_k = new_k.replace("attn1.to_out.0", "self_attention.o_proj")
                    
                    # MLP
                    # Checkpoint: ff.net.0.proj (Standard MLP Proj 1), ff.net.2 (Standard MLP Proj 2)
                    # DiaModel (SwiGLU): mlp.gate_up_proj, mlp.down_proj
                    # gate_up_proj expects [2*Intermediate, Hidden]
                    # Checkpoint provides [Intermediate, Hidden]
                    if "ff.net.0.proj" in k:
                        new_k = new_k.replace("ff.net.0.proj", "mlp.gate_up_proj")
                        # Handle shape mismatch for SwiGLU
                        # DiaModel likely expects cat(gate, up).
                        # We use checkpoint weights for UP, and random for GATE.
                        # Target Shape: [8192, 1024]. Checkpoint: [4096, 1024].
                        w_up = v
                        # Initialize gate with Kaiming uniform or similar (using torch.randn for simplicity here)
                        # We scale it to be small to let the UP path dominate early on?
                        # Or standard initialization.
                        w_gate = torch.randn_like(w_up) * 0.02
                        
                        # Concatenate [gate, up] (Common implementation order usually gate then up, or vice versa)
                        # We'll assume gate|up.
                        v = torch.cat([w_gate, w_up], dim=0)
                        
                    elif "ff.net.2" in k:
                        new_k = new_k.replace("ff.net.2", "mlp.down_proj")
                        # down_proj matches [Out, Intermediate] -> [1024, 4096]. No change needed.
                    
                    # Norms
                    new_k = new_k.replace("norm1", "pre_sa_norm")
                    # norm2 might be cross-attn norm? Checkpoint usually only has self-attn in backbone
                    new_k = new_k.replace("norm3", "pre_mlp_norm")

                # Embeddings
                if "vl_self_attention_model.pos_embed" in new_k:
                    # Dia decoder might not use learnable pos embed if using RoPE, but we'll try
                    pass 

                new_sd[new_k] = v
            
            state_dict = new_sd
            logger.info(f"State dict prepared with {len(state_dict)} keys.")
        except Exception as e:
            logger.error(f"Failed to fix state dict: {e}")
            state_dict = None

    # We use AutoModel because NitroGen likely doesn't fit a standard task head class like CausalLM perfectly
    logger.info("Instantiating model from config...")
    config = AutoConfig.from_pretrained(
        local_model_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Patch Config to match Checkpoint (Small vs Base/Large mismatch)
    # The checkpoint has hidden_size=1024, layers=4
    # The default config has hidden_size=2048, layers=18
    if hasattr(config, "decoder_config"):
        logger.info("Patching decoder_config to match checkpoint weights...")
        config.decoder_config.hidden_size = 1024
        config.decoder_config.intermediate_size = 4096 # Corrected from checkpoint (4096)
        config.decoder_config.num_hidden_layers = 4
        config.decoder_config.num_attention_heads = 8 # 1024 / 128 = 8
        config.decoder_config.num_key_value_heads = 8 # Checkpoint has 1024 dim for K/V, so 8 heads * 128 dim. No GQA.
        
        # Also patch encoder if strictly tied, though we prioritize decoder loading
        # config.encoder_config.hidden_size = 1024 
    
    model = AutoModel.from_config(
        config,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Move to correct dtype
    model.to(dtype=torch_dtype)

    # Load fixed weights if available
    if state_dict is not None:
        logger.info("Loading fixed state_dict into model...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # Optional: Log missing keys to see if critical ones are missing
        if len(missing) > 0:
            logger.debug(f"Missing keys: {missing[:10]}...")
            
    # Add Task Head (Linear Projection from Hidden -> Horizon * ActionDim)
    # Hidden 1024 -> 16 * 21 = 336
    # This head needs to be trained.
    logger.info("Adding 'action_head' linear layer to model...")
    model.action_head = torch.nn.Linear(1024, 16 * 21)
    # Initialize head
    model.action_head.weight.data.normal_(mean=0.0, std=0.02)
    model.action_head.bias.data.zero_()
    # Ensure head is float32 for stability and to prevent "unscale FP16 gradients" error.
    # Trainer with fp16=True usually expects unscaled gradients in FP32 from the optimizer logic,
    # but having the module in FP32 helps ensure consistent Master Weights behavior for this custom head.
    model.action_head.to(dtype=torch.float32)

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
            modules_to_save=["action_head"], # Train the head fully
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 4. Prepare Dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    try:
        raw_dataset = datasets.load_dataset(
            data_args.dataset_name, split="train"
        )
    except Exception as e:
        logger.warning(f"Failed to load dataset normally: {e}. Trying explicit cache/path if needed.")
        raise e

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
