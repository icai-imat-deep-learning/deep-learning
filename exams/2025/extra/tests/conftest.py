"""
This module contains the code for the fixtures
"""

# 3pps
import torch
import pytest

# Own modules
from tests.utils import add_seed, set_seed


@pytest.fixture(params=[*add_seed((32, 6, 16, 16)), *add_seed((16, 18, 32, 32))])
def inputs_2d(request) -> torch.Tensor:
    """
    This function is a fixture to define example random inputs.

    Args:
        request: Argument containing the introduced arguments.

    Returns:
        Inputs tensor. Dimensions: [batch, channels, height, width].
    """

    # Get parameters
    batch_size: int
    num_channels: int
    height: int
    width: int
    seed: int
    batch_size, num_channels, height, width, seed = request.param

    # Set seed
    set_seed(seed)

    # Define inputs
    inputs: torch.Tensor = (
        torch.rand(batch_size, num_channels, height, width).uniform_(-10, 10).double()
    )

    return inputs
