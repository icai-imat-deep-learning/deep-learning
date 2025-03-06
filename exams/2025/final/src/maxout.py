"""
This module contains the code for Hardshrink.
"""

# Standard libraries
from typing import Any

# 3pps
import torch


class MaxOut(torch.nn.Module):
    def __init__(self, num_units: int, input_dim: int, output_dim: int) -> None:
        # Call super class
        super().__init__()

        # Set attributes
        self.num_units = num_units
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand((self.num_units, output_dim, input_dim), dtype=torch.double)
        )

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Repeat inputs
        outputs = inputs.view(inputs.shape[0], 1, 1, inputs.shape[1])
        outputs = outputs.repeat(1, self.num_units, 1, 1)

        # Get weight
        weight = self.weight.view(1, *self.weight.shape)
        weight = weight.repeat(inputs.shape[0], 1, 1, 1)

        # Compact 1 and 2 dimension
        outputs = outputs.view(-1, *outputs.shape[2:])
        weight = weight.view(-1, *weight.shape[2:])
        weight = weight.transpose(1, 2)

        # Compute outputs
        outputs = torch.bmm(outputs, weight)

        # Compute final shape
        outputs = outputs.view(inputs.shape[0], self.num_units, -1)

        # Compute max accross units
        outputs = torch.amax(outputs, dim=1)

        return outputs
