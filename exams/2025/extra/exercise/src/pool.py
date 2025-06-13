"""
This module contains the code to implement CustomMaxPool2d.
"""

# Standard libraries
from typing import Any

# 3pps
import torch
import torch.nn.functional as F


class CustomMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, kernel_size: int, num_groups: int
    ) -> torch.Tensor:
        """
        This is the forward method of the CustomMaxPool2d.

        Args:
            ctx: Context for saving elements for the backward.
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, number of groups,
                height - kernel size + 1,
                width - kernel size + 1].
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        """
        This method implements the backward pass of the layer.

        Args:
            grad_outputs: Outputs gradients. Dimensions: [batch size,
                number of groups, height - kernel size + 1,
                width - kernel size + 1].

        Returns:
            Gradients of the inputs. Dimensions: [batch size,
                number of channels, height, width].
            None.
            None.
        """

        # TODO


class CustomMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: int, num_groups: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            kernel_size: Kernel size.
            num_groups: Number of groups.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        # Set function
        self.fn = CustomMaxPool2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, channels,
                height, width].

        Returns:
            Outputs tensor. Dimensions: [batch, channels, ].
        """

        # Get outputs
        outputs: torch.Tensor = self.fn(inputs, self.kernel_size, self.num_groups)

        return outputs
