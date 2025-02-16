"""
This module contains the code for testing the Hardshrink.
"""

# 3pps
import torch
import pytest

# Own libraries
from src.hardshrink import Hardshrink
from tests.utils import set_seed


@pytest.mark.order(2)
def test_hardshrink_forward() -> None:
    """
    This function is to test the Hardshrink forward pass.

    Returns:
        None.
    """

    # Define modules
    module: torch.nn.Module = Hardshrink()
    module_torch: torch.nn.Module = torch.nn.Hardshrink()

    # Iter over seeds
    for seed in range(10):
        # Set seed
        set_seed(seed)

        # Define inputs
        inputs: torch.Tensor = torch.rand(64, 6, 32, 32).uniform_(-10, 10)

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

    # Define inputs
    inputs = torch.zeros(64, 6, 32, 32)

    # Compute outputs
    outputs = module(inputs)
    outputs_torch = module_torch(inputs)

    # Check outputs value
    assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value at zero"

    return None


@pytest.mark.order(3)
def test_hardshrink_backward() -> None:
    """
    This function is to test the Hardshrink backward pass.

    Returns:
        None.
    """

    # Define modules
    module: torch.nn.Module = Hardshrink()
    module_torch: torch.nn.Module = torch.nn.Hardshrink()

    # Iter over seeds
    for seed in range(10):
        # Set seed
        set_seed(seed)

        # Define inputs
        inputs: torch.Tensor = (
            torch.rand(64, 6, 32, 32).uniform_(-10, 10).requires_grad_(True)
        )

        # Compute outputs backward
        outputs: torch.Tensor = module(inputs)
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # Compute torch outputs backward
        outputs_torch: torch.Tensor = module_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch = inputs.grad

        # Check shape
        assert grad_inputs.shape == grad_inputs_torch.shape, (
            f"Incorrect gradients inputs shape, expected {grad_inputs_torch.shape} "
            f"and got {grad_inputs.shape}"
        )

        # Check outputs value
        assert torch.allclose(
            grad_inputs, grad_inputs_torch
        ), "Incorrect gradients inputs value"

    # Define inputs
    inputs = torch.zeros(64, 6, 32, 32)

    # Compute outputs backward
    outputs = module(inputs)
    outputs.sum().backward()
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_inputs = inputs.grad.clone()

    # Compute torch outputs backward
    outputs_torch = module_torch(inputs)
    inputs.grad.zero_()
    outputs_torch.sum().backward()
    if inputs.grad is None:
        assert False, "Gradients not returned, none value detected"
    grad_inputs_torch = inputs.grad

    # Check outputs value
    assert torch.allclose(
        grad_inputs, grad_inputs_torch
    ), "Incorrect gradients inputs value at zero"

    return None
