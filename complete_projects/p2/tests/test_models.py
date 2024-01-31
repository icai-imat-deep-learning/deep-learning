# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.models import ReLU, Linear, Conv2d, Block
from src.utils import set_seed

# set seed and device
set_seed(42)


@pytest.mark.order(1)
def test_relu() -> None:
    """
    This function is the test for the relu function.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs.requires_grad_(True)

    # define linear
    model: torch.nn.Module = ReLU()

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "gradients not returned, none value detected"
    inputs_grad: torch.Tensor = inputs.grad

    # define torch linear
    model = torch.nn.ReLU()

    # compute outputs and backward
    outputs = model(inputs)
    model.zero_grad()
    inputs.grad.zero_()
    outputs.sum().backward()

    # get grads values
    if inputs.grad is None:
        assert False, "gradients not returned, none value detected"
    inputs_grad_torch: torch.Tensor = inputs.grad

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect calculation of inputs gradients"

    return None


@pytest.mark.order(2)
def test_linear() -> None:
    """
    This function is the test for the linear model.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs.requires_grad_(True)

    # define linear
    model: torch.nn.Module = Linear(30, 10)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads values
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "gradients not returned, none value detected"
    grad_weight: torch.Tensor = model.weight.grad
    grad_bias: torch.Tensor = model.bias.grad
    inputs_grad: torch.Tensor = inputs.grad

    # define torch linear
    model = torch.nn.Linear(30, 10)

    # compute outputs and backward
    outputs = model(inputs)
    model.zero_grad()
    inputs.grad.zero_()
    outputs.sum().backward()

    # get grads values
    if model.weight.grad is None or model.bias.grad is None or inputs.grad is None:
        assert False, "gradients not returned, none value detected"
    grad_weight_torch: torch.Tensor = model.weight.grad
    grad_bias_torch: torch.Tensor = model.bias.grad
    inputs_grad_torch: torch.Tensor = inputs.grad

    # check weights grads
    assert (
        grad_weight != grad_weight_torch
    ).sum().item() == 0, "Incorrect calculation of weights gradients"

    # check bias grads
    assert (
        grad_bias != grad_bias_torch
    ).sum().item() == 0, "Incorrect calculation of bias gradients"

    # check inputs grads
    assert (
        inputs_grad != inputs_grad_torch
    ).sum().item() == 0, "Incorrect calculation of inputs gradients"

    return None


@pytest.mark.order(3)
def test_conv() -> None:
    """
    This function is the test for the conv model.
    """

    # define inputs
    inputs: torch.Tensor = torch.rand(64, 3, 32, 32).double()
    inputs.requires_grad_(True)

    # define conv

    model: torch.nn.Module = Conv2d(3, 10, 7)

    # compute outputs and backward
    outputs = model(inputs)
    outputs.sum().backward()

    # get grads
    grad_inputs: torch.Tensor = inputs.grad
    grad_weight: torch.Tensor = model.weight.grad
    grad_bias: torch.Tensor = model.bias.grad

    # define conv
    model_torch = torch.nn.Conv2d(3, 10, 7)
    model_torch.weight = torch.nn.Parameter(model.weight.detach().clone())
    model_torch.bias = torch.nn.Parameter(model.bias.detach().clone())

    # compute outputs and backward
    outputs_torch = model_torch(inputs)
    inputs.grad.zero_()
    outputs_torch.sum().backward()

    # get grads
    grad_inputs_torch: torch.Tensor = inputs.grad
    grad_weight_torch: torch.Tensor = model_torch.weight.grad
    grad_bias_torch: torch.Tensor = model_torch.bias.grad

    assert (outputs != outputs_torch).sum() == 0, "Incorrect forward"

    assert (grad_inputs != grad_inputs_torch).sum() == 0, "Incorrect inputs gradients"

    assert (grad_weight != grad_weight_torch).sum() == 0, "Incorrect weights gradient"

    assert (grad_bias != grad_bias_torch).sum() == 0, "Incorrect bias gradients"

    return None


# @pytest.mark.order(1)
# @pytest.mark.parametrize(
#     "input_channels, output_channels, stride", [(3, 10, 2), (1, 20, 4)]
# )
# def test_block(input_channels: int, output_channels: int, stride: int) -> None:
#     """
#     This is a test for the Block class.

#     Args:
#         input_channels: inut channels.
#         output_channels: output channels.
#         stride: stride.
#     """

#     # define block
#     block: torch.nn.Module = Block(input_channels, output_channels, stride)

#     # check sequential object
#     assert isinstance(
#         list(block.children())[0], torch.nn.Sequential
#     ), "Layers not encapsulated inside sequential"

#     # check length
#     assert len(list(block.children())[0]) == 6, "Incorrect length "

#     if isinstance(list(block.children())[0], torch.nn.Sequential):
#         sequential: torch.nn.Sequential = list(block.children())[0]

#         # check stride 1 in first conv layer
#         assert list(block.children())[0][0].stride == (
#             1,
#             1,
#         ), "Incorrect stride in first conv layer"

#         # check stride 1 in first conv layer
#         assert list(block.children())[0][2].stride == (
#             2,
#             2,
#         ), "Incorrect stride in first conv layer"

#     # define input and compute output
#     example_input: torch.Tensor = torch.rand(64, input_channels, 224, 224)
#     example_output: torch.Tensor = block(example_input)

#     # check type of object
#     assert isinstance(example_output, torch.Tensor), "Incorrect output object type"

#     # check dimensions
#     assert example_output.shape == (
#         64,
#         output_channels,
#         224 // stride,
#         224 // stride,
#     ), "Incorrect shape of output object"

#     return None
