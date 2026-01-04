import pytest
import torch
from unittest.mock import MagicMock
from src.train import DiTTrainer

def test_compute_loss_mse():
    """Test that compute_loss calculates MSE correctly when model returns logits."""
    # Mock model
    model = MagicMock()
    
    # Inputs
    batch_size = 2
    horizon = 4
    action_dim = 21
    
    # Random predictions and targets
    logits = torch.randn(batch_size, horizon, action_dim)
    labels = torch.randn(batch_size, horizon, action_dim)
    
    # Model output mock
    model_output = MagicMock()
    model_output.loss = None
    model_output.logits = logits
    model.return_value = model_output
    
    # Trainer instance
    trainer = DiTTrainer(model=model, args=MagicMock(), train_dataset=None, eval_dataset=None)
    
    inputs = {
        "pixel_values": torch.randn(batch_size, 3, 224, 224),
        "actions": labels
    }
    
    loss = trainer.compute_loss(model, inputs)
    
    # Expected MSE
    expected_loss = torch.nn.functional.mse_loss(logits, labels)
    
    # Verify
    assert torch.isclose(loss, expected_loss)
    
def test_compute_loss_internal():
    """Test that compute_loss uses internal model loss if available."""
    model = MagicMock()
    
    # Model output mock with internal loss
    internal_loss = torch.tensor(0.5)
    model_output = MagicMock()
    model_output.loss = internal_loss
    model.return_value = model_output
    
    trainer = DiTTrainer(model=model, args=MagicMock(), train_dataset=None, eval_dataset=None)
    
    inputs = {
        "pixel_values": torch.randn(2, 3, 224, 224),
        "actions": torch.randn(2, 4, 21)
    }
    
    loss = trainer.compute_loss(model, inputs)
    
    assert loss == internal_loss
