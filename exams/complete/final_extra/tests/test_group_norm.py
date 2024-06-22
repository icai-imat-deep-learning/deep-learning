# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.group_norm import GroupNorm
from src.utils import set_seed


@pytest.mark.order(1)
def test_group_norm() -> None:
    """
    This function is the test for the for

    Returns:
        _description_
    """

    # loop with different seeds
    for seed in range(10):
        # define inputs
        scale = torch.randint(1, 10, (1,)).double()
        bias = torch.randint(-10, 10, (1,)).double()
        inputs: torch.Tensor = (
            torch.rand(64, 6, 32, 32, dtype=torch.double) * scale + bias
        )

        # define models
        set_seed(seed)
        model = GroupNorm(2, 6, affine=False, eps=0, dtype=torch.double)
        set_seed(seed)
        model_torch = torch.nn.GroupNorm(2, 6, affine=False, eps=0, dtype=torch.double)

        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-3), "Incorrect outputs"

    return None
