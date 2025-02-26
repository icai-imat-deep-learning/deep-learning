""" """

# 3pps
import torch
import pytest

# Own modules
from src.maxout import MaxOut
from tests.utils import MaxOutLoop


@pytest.mark.parametrize("num_units, output_dim", [(2, 1), (10, 10)])
def test_maxout(num_units: int, output_dim: int, inputs_linear: torch.Tensor) -> None:
    """
    _summary_

    Args:
        num_units: _description_
        output_dim: _description_
        inputs_linear: _description_

    Returns:
        _description_
    """

    # Define maxout layers
    maxout = MaxOut(num_units, inputs_linear.shape[1], output_dim)
    maxout_loop = MaxOutLoop(num_units, inputs_linear.shape[1], output_dim)

    # Compute
    maxout_loop.load_parameters(maxout)

    # Compute outputs
    outputs: torch.Tensor = maxout(inputs_linear)
    outputs_loop: torch.Tensor = maxout_loop(inputs_linear)

    # Check shape
    assert outputs.shape == outputs_loop.shape, (
        f"Incorrect outputs shape, expected {outputs_loop.shape} and got "
        f"{outputs.shape}"
    )

    # Check outputs
    assert torch.allclose(outputs, outputs_loop), "Incorrect forward prediction"

    return None
