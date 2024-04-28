# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.maxout import Maxout
from src.utils import parameters_to_double, set_seed

# set seed
set_seed(42)


@pytest.mark.order(6)
@torch.no_grad()
def test_maxout_forward() -> None:
    """
    This function test the maxout.
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        torch.zeros_like(model.weights_first),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    model_torch: torch.nn.Module = torch.nn.ReLU()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the relu"

    # define maxout
    model = Maxout(32, 32)
    model.set_parameters(
        -1 * torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_first),
        torch.eye(model.weights_first.shape[0]),
        torch.zeros_like(model.bias_second),
    )

    # compute outputs and backward
    outputs = model(inputs)

    # compute outputs
    outputs_torch = torch.abs(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward simulating the absolute value"

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    model_torch.linear1.weight.data = model.weights_first.clone()
    model_torch.linear1.bias.data = model.bias_first.clone()
    model_torch.linear2.weight.data = model.weights_second.clone()
    model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward of implementation with nn"

    return None


@pytest.mark.order(7)
def test_maxout_backward() -> None:
    """
    _summary_

    Returns:
        _description_
    """

    # set seed
    set_seed(42)

    # define inputs
    inputs: torch.Tensor = torch.FloatTensor(64, 32).uniform_(-10, 10)
    inputs[0, 0] = 0

    # define maxout
    model = Maxout(32, 20)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward

    # define torch relu
    class MaxoutTorch(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            # call super class constructor
            super().__init__()

            self.linear1 = torch.nn.Linear(input_dim, output_dim)
            self.linear2 = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs1: torch.Tensor = self.linear1(inputs)
            outputs2: torch.Tensor = self.linear2(inputs)
            outputs = torch.maximum(outputs1, outputs2)

            return outputs

    # compute outputs
    model_torch = MaxoutTorch(32, 20)
    model_torch.linear1.weight.data = model.weights_first.clone()
    model_torch.linear1.bias.data = model.bias_first.clone()
    model_torch.linear2.weight.data = model.weights_second.clone()
    model_torch.linear2.bias.data = model.bias_second.clone()

    # compute outputs
    outputs_torch = model_torch(inputs)

    # check outputs
    assert (
        outputs != outputs_torch
    ).sum().item() == 0, "Incorrect forward of implementation with nn"

    return None
