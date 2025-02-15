"""
This module contains the code for testing the GroupNorm.
"""

# 3pps
import torch
import pytest

# Own libraries
from src.group_norm import GroupNorm
from tests.utils import set_seed


@pytest.mark.order(1)
def test_group_norm() -> None:
    """
    This function is to test the GroupNorm.

    Returns:
        None.
    """
    
    for seed in range(10):
        # Set seed
        set_seed(seed)

        # Define inputs
        inputs: torch.Tensor = torch.rand(64, 6, 32, 32).uniform_(-10, 10)

        # Define modules
        module: torch.nn.Module = GroupNorm(2, 6)
        module_torch: torch.nn.Module = torch.nn.GroupNorm(2, 6, affine=False)

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs_torch: torch.Tensor = module_torch(inputs)

        # Check shape
        assert outputs.shape == outputs_torch.shape, (
            f"Incorrect output shape, expected {outputs_torch.shape} and got "
            f"{outputs.shape}"
        )

        # Check outputs value
        assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value"

    return None
