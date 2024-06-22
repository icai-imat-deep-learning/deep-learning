# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.max_pool import unfold_max_pool, MaxPool2d
from src.utils import set_seed


@pytest.mark.order(2)
def test_unfold_max_pool() -> None:
    """
    This function is the test for the unfold1d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 3, 32, 32)

    # unfold inputs
    inputs_unfolded: torch.Tensor = unfold_max_pool(inputs, 4, 1, 0)

    # check dimensions
    assert inputs_unfolded.shape[:2] == (64 * 3, 16), "Incorrect shape of unfold"

    # check values
    assert (
        inputs[0, 0, :4, :4].reshape(-1) != inputs_unfolded[0, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 1, :4, :4].reshape(-1) != inputs_unfolded[1, :, 0]
    ).sum().item() == 0, "Incorrect values of unfold"
    assert (
        inputs[0, 0, :4, 1:5].reshape(-1) != inputs_unfolded[0, :, 1]
    ).sum().item() == 0, "Incorrect values of unfold"

    return None


@pytest.mark.order(2)
def test_max_pool_forward() -> None:
    # loop with different seeds
    for seed in range(10):
        # define inputs
        inputs: torch.Tensor = torch.rand(64, 6, 32, 32)

        # define models
        set_seed(seed)
        model = MaxPool2d(4, stride=1)
        set_seed(42)
        model_torch = torch.nn.MaxPool2d(4, stride=1)

        # compute outputs
        outputs = model(inputs)
        outputs_torch = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-10), "Incorrect outputs"

    return None
