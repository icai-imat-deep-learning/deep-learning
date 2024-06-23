# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.softshrink import Softshrink
from src.utils import set_seed


@pytest.mark.order(3)
def test_softshrink_forward() -> None:
    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(64, 32)

        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        # define models
        model = Softshrink()
        model_torch = torch.nn.Softshrink()

        # compute outputs
        outputs: torch.Tensor = model(inputs)
        outputs_torch: torch.Tensor = model_torch(inputs)

        # check output size
        assert (
            outputs.shape == outputs_torch.shape
        ), f"Incorrect outputs shape, expected {outputs_torch.shape}, got {outputs.shape}"

        # check outputs
        assert torch.allclose(outputs, outputs_torch, atol=1e-4), "Incorrect outputs"

    return None


@pytest.mark.order(4)
def test_softshrink_backward() -> None:
    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.rand(64, 32)

        inputs[0, 0] = 0.5
        inputs[0, 1] = -0.5

        inputs.requires_grad_(True)

        # define models
        model = Softshrink()
        model_torch = torch.nn.Softshrink()

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
            grad_inputs, grad_inputs_torch, atol=1e-4
        ), "Incorrect outputs"

    return None
