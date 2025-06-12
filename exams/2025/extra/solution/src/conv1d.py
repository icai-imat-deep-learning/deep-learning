"""
This module contains the code for the Conv1d implementation.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Get output size and define outputs
        output_size: int = inputs.shape[2] - weight.shape[2] + 1
        outputs: torch.Tensor = torch.zeros(
            (inputs.shape[0], weight.shape[0], output_size), dtype=torch.double
        )

        # Iter over spatial dimension
        for i in range(output_size):
            outputs[:, :, i] = (
                inputs[:, :, i : i + weight.shape[2]].unsqueeze(1) * weight.unsqueeze(0)
            ).sum(dim=(2, 3))

        # Save for backward
        ctx.save_for_backward(inputs, weight)

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get tensors from forward
        inputs, weight = ctx.saved_tensors

        #
        grad_inputs = torch.zeros_like(inputs)

        # Iter over spatial dimension
        for i in range(grad_outputs.shape[2]):
            grad_inputs[:, :, i : i + weight.shape[2]] += (
                grad_outputs[:, :, i].unsqueeze(1).unsqueeze(-1)
                * weight.unsqueeze(0).permute(0, 2, 1, 3)
            ).sum(dim=2)
            
        grad_weight

        return grad_inputs, grad_weight


class Conv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """
        _summary_

        Args:
            in_channels: _description_
            out_channels: _description_
            kernel_size: _description_

        Returns:
            _description_
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.kernel = torch.nn.Parameter(
            torch.rand((out_channels, in_channels, kernel_size), dtype=torch.double)
        )

        # Set function
        self.fn = Conv1dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        _summary_

        Args:
            inputs: _description_

        Returns:
            _description_
        """

        # Compute outputs
        outputs: torch.Tensor = self.fn(inputs, self.kernel)

        return outputs
