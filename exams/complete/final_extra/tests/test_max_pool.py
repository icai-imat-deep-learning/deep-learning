# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.max_pool import unfold_max_pool_2d, MaxPool2d
from src.utils import set_seed


@pytest.mark.order(2)
def test_unfold_max_pool_2d() -> None:
    """
    This function is the test for the unfold1d.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 3, 32, 32)

    # unfold inputs
    inputs_unfolded: torch.Tensor = unfold_max_pool_2d(inputs, 4, 1, 0)

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


@pytest.mark.order(4)
def test_max_pool_forward() -> None:
    # loop with different seeds
    for seed in range(10):
        # define inputs
        inputs: torch.Tensor = torch.rand(64, 3, 32, 32)

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


@pytest.mark.order(5)
def test_max_pool_backward() -> None:
    # loop with different seeds
    for seed in range(10):
        # set seed
        set_seed(seed)

        # define inputs
        inputs: torch.Tensor = torch.rand(64, 3, 32, 32)
        inputs.requires_grad_(True)

        # define models
        model = MaxPool2d(4, stride=1)
        model_torch = torch.nn.MaxPool2d(4, stride=1)

        # compute backward of our maxpool
        outputs = model(inputs)
        if inputs.grad is not None:
            inputs.grad.zero_()
        outputs.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs: torch.Tensor = inputs.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        inputs.grad.zero_()
        outputs_torch.sum().backward()
        if inputs.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_inputs_torch: torch.Tensor = inputs.grad.clone()

        # check output size
        assert (
            grad_inputs.shape == grad_inputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(
            grad_inputs, grad_inputs_torch, atol=1e-10
        ), "Incorrect outputs"

    return None
