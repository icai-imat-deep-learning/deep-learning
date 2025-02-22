"""
This module contains the code for testing the Hardshrink.
"""

# 3pps
import torch
import pytest

# Own libraries
from src.hardshrink import Hardshrink


class TestHardShrinnk:
    @pytest.mark.order(2)
    def test_hardshrink_forward(
        self, inputs: torch.Tensor, inputs_zero: torch.Tensor
    ) -> None:
        """
        This function is to test the Hardshrink forward pass.

        Returns:
            None.
        """

        # Define modules
        module: torch.nn.Module = Hardshrink()
        module_torch: torch.nn.Module = torch.nn.Hardshrink()

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

        # Compute outputs
        outputs = module(inputs_zero)
        outputs_torch = module_torch(inputs_zero)

        # Check outputs value
        assert torch.allclose(outputs, outputs_torch), "Incorrect outputs value at zero"

        return None

    @pytest.mark.order(3)
    def test_hardshrink_backward(
        self, inputs: torch.Tensor, inputs_zero: torch.Tensor
    ) -> None:
        """
        This function is to test the Hardshrink backward pass.

        Returns:
            None.
        """

        # Activate gradients
        inputs.requires_grad_(True)
        inputs_zero.requires_grad_(True)

        # Define modules
        module: torch.nn.Module = Hardshrink()
        module_torch: torch.nn.Module = torch.nn.Hardshrink()

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

        # Compute outputs backward
        outputs = module(inputs_zero)
        outputs.sum().backward()
        if inputs_zero.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs = inputs_zero.grad.clone()

        # Compute torch outputs backward
        outputs_torch = module_torch(inputs_zero)
        inputs_zero.grad.zero_()
        outputs_torch.sum().backward()
        if inputs_zero.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch = inputs_zero.grad

        # Check outputs value
        assert torch.allclose(
            grad_inputs, grad_inputs_torch
        ), "Incorrect gradients inputs value at zero"

        return None
