from unittest.mock import MagicMock

import pytest
import torch

from src.train import DiTTrainer


def test_compute_loss_delegation():
    """Test that compute_loss delegates to model's internal loss calculation."""
    model = MagicMock(spec=torch.nn.Module)
    model.tp_size = 1
    model._modules = {}

    # Internal loss to expect
    internal_loss = torch.tensor(0.5)

    # When model(inputs) is called, it should return a dict-like object containing 'loss'
    # We can use a real dict for simplicity, as MagicMock return_value can be anything
    model.return_value = {"loss": internal_loss}

    # Mock args
    args = MagicMock()
    args.eval_strategy = "no"
    # Basic args to satisfy Trainer init
    args.seed = 42
    args.deepspeed = None
    args.deepspeed_plugin = None
    args.max_steps = 0
    args.num_train_epochs = 0
    args.get_process_log_level = MagicMock(return_value=20)
    args.use_liger_kernel = False
    args.label_smoothing_factor = 0
    args.fsdp = []
    args.fsdp_config = {"xla": False, "xla_fsdp_v2": False}
    # Mock accelerator config stuff
    args.accelerator_config.to_dict.return_value = {
        "split_batches": False,
        "dispatch_batches": None,
        "even_batches": True,
        "use_seedable_sampler": True,
        "non_blocking": False,  # Important for verify_args
        "gradient_accumulation_kwargs": None,
    }

    # To avoid "ValueError: non_blocking must be True..." or similar deepspeed/accelerator checks,
    # we usually need to be careful with args mocks.
    # But DiTTrainer doesn't do much extra init logic that varies from Trainer.

    trainer = DiTTrainer(model=model, args=args, train_dataset=None, eval_dataset=None)

    inputs = {
        "pixel_values": torch.randn(2, 3, 224, 224),
        "actions": torch.randn(2, 4, 21),
    }

    loss = trainer.compute_loss(model, inputs)

    assert loss == internal_loss

    # Verify model was called with inputs
    model.assert_called_with(inputs)
