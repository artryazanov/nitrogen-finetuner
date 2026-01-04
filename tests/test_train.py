from unittest.mock import MagicMock

import pytest
import torch

from src.train import DiTTrainer


def test_compute_loss_mse():
    """Test that compute_loss calculates MSE correctly when model returns logits."""
    # Mock model
    model = MagicMock(spec=torch.nn.Module)
    model.tp_size = 1
    model._modules = {}
    model.config = MagicMock()
    model.config.bos_token_id = 0

    # Inputs
    batch_size = 2
    horizon = 4
    action_dim = 21

    # Random predictions and targets
    logits = torch.randn(batch_size, horizon, action_dim)
    labels = torch.randn(batch_size, horizon, action_dim)
    
    # Mock action_head
    # compute_loss calls: output = action_head(hidden); output.view(B, H, A)
    # So action_head should return (B, 1, H*A) or similar such that view works.
    # We'll just make it return the reshaped logits directly if we mock the side effect, 
    # OR better: make return value (B, 1, H*A).
    flat_logits = logits.reshape(batch_size, 1, -1)
    
    action_head = MagicMock()
    action_head.return_value = flat_logits
    # Mock weight.dtype for the .to() call in compute_loss
    action_head.weight = MagicMock()
    action_head.weight.dtype = torch.float32
    model.action_head = action_head

    # Model output mock
    model_output = MagicMock()
    model_output.loss = None
    # For compute_loss, outputs[0] is hidden states.
    # We can pass dummy hidden states.
    model_output.__getitem__ = MagicMock(return_value=torch.randn(batch_size, 1, 1024))
    model.return_value = model_output

    # Mock args to avoid eval_strategy check
    args = MagicMock()
    args.eval_strategy = "no"
    args.seed = 42
    args.deepspeed_plugin = None
    args.parallelism_config = None
    args.accelerator_config.gradient_accumulation_kwargs = None
    args.accelerator_config.to_dict.return_value = {
        "split_batches": False,
        "dispatch_batches": None,
        "even_batches": True,
        "use_seedable_sampler": True,
        "non_blocking": False,
        "gradient_accumulation_kwargs": None,
    }
    args.get_process_log_level.return_value = 20
    args.use_liger_kernel = False
    args.max_steps = 0
    args.num_train_epochs = 0
    args.label_smoothing_factor = 0
    args.fsdp = []
    args.fsdp_config = {"xla": False, "xla_fsdp_v2": False}

    # Trainer instance
    trainer = DiTTrainer(model=model, args=args, train_dataset=None, eval_dataset=None)

    inputs = {"pixel_values": torch.randn(batch_size, 3, 224, 224), "actions": labels}

    loss = trainer.compute_loss(model, inputs)

    # Expected MSE
    expected_loss = torch.nn.functional.mse_loss(logits, labels)

    # Verify
    assert torch.isclose(loss, expected_loss)


def test_compute_loss_internal():
    """Test that compute_loss uses internal model loss if available."""
    model = MagicMock(spec=torch.nn.Module)
    model.tp_size = 1
    model._modules = {}
    model.config = MagicMock()
    model.config.bos_token_id = 0

    # Model output mock with internal loss
    internal_loss = torch.tensor(0.5)
    model_output = MagicMock()
    model_output.loss = internal_loss
    model.return_value = model_output

    # Mock args to avoid eval_strategy check
    args = MagicMock()
    args.eval_strategy = "no"
    args.seed = 42
    args.deepspeed_plugin = None
    args.parallelism_config = None
    args.accelerator_config.gradient_accumulation_kwargs = None
    args.accelerator_config.to_dict.return_value = {
        "split_batches": False,
        "dispatch_batches": None,
        "even_batches": True,
        "use_seedable_sampler": True,
        "non_blocking": False,
        "gradient_accumulation_kwargs": None,
    }
    args.get_process_log_level.return_value = 20
    args.use_liger_kernel = False
    args.max_steps = 0
    args.num_train_epochs = 0
    args.label_smoothing_factor = 0
    args.fsdp = []
    args.fsdp_config = {"xla": False, "xla_fsdp_v2": False}

    trainer = DiTTrainer(model=model, args=args, train_dataset=None, eval_dataset=None)

    inputs = {
        "pixel_values": torch.randn(2, 3, 224, 224),
        "actions": torch.randn(2, 4, 21),
    }

    loss = trainer.compute_loss(model, inputs)

    assert loss == internal_loss
