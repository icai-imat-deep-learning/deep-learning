"""
This module contains the code to test the ModePool.
"""

# 3pps
import torch
import pytest

# Own modules
from src.pool import CustomMaxPool2d
from tests.utils import TestCustomMaxPool2d


@pytest.mark.parametrize("kernel_size, num_groups", [(3, 1)])
class TestMaxPool2d:
    @torch.no_grad()
    def test_forward(
        self, kernel_size: int, num_groups: int, inputs_2d: torch.Tensor
    ) -> None:
        # Rename fixture
        inputs: torch.Tensor = inputs_2d

        # Define modules
        module: torch.nn.Module = CustomMaxPool2d(kernel_size, num_groups)
        module_test: torch.nn.Module = TestCustomMaxPool2d(kernel_size, num_groups)

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs_test: torch.Tensor = module_test(inputs)

        # Check values
        torch.allclose(outputs, outputs_test)

        return None

    def test_backward(
        self, kernel_size: int, num_groups: int, inputs_2d: torch.Tensor
    ) -> None:
        # Rename fixture
        inputs: torch.Tensor = inputs_2d
        inputs.requires_grad_(True)

        # Define modules
        module: torch.nn.Module = CustomMaxPool2d(kernel_size, num_groups)
        module_test: torch.nn.Module = TestCustomMaxPool2d(kernel_size, num_groups)

        # Compute outputs
        outputs: torch.Tensor = module(inputs)
        outputs.sum().backward()
        if inputs.grad is None:
            raise RuntimeError("Error in gradient computation")
        gradients: torch.Tensor = inputs.grad.clone()
        inputs.grad.zero_()

        # Compute torch gradients
        outputs = module_test(inputs)
        outputs.sum().backward()
        if inputs.grad is None:
            raise RuntimeError("Error in gradient computation")
        gradients_test: torch.Tensor = inputs.grad.clone()
        inputs.grad.zero_()

        # Check values
        torch.allclose(gradients, gradients_test)

        return None
