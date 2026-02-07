"""
This module contains the tests for the src.models.
"""

# 3pps
import pytest
import torch

# Own modules
from src.models import Linear, MyModel, ReLU


@pytest.mark.order(2)
def test_relu() -> None:
    """
    This is the test for the ReLU class.

    Raises:
        RuntimeError: Error with gradient computation in test.
        RuntimeError: Error with gradient computation in test.
        RuntimeError: Error with gradient computation in test.
    """

    # Define relu class
    relu: torch.nn.Module = ReLU()

    # Check positive forward
    example_input: torch.Tensor = (
        torch.FloatTensor(3, 3).uniform_(0.1, 10).requires_grad_(True)
    )
    example_output: torch.Tensor = relu(example_input)
    assert (
        example_input != example_output
    ).sum() == 0, "Incorrect forward for positive numbers"

    # Check positive backward
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (
        example_input.grad != 1
    ).sum() == 0, "Incorrect backward for positive numbers"

    # Check negative transformation
    example_input = torch.FloatTensor(3, 3).uniform_(-10, -0.1).requires_grad_(True)
    example_output = relu(example_input)
    assert (example_output != 0).sum() == 0, "Incorrect forward for negative numbers"

    # Check negative backward
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (
        example_input.grad != 0
    ).sum() == 0, "Incorrect backward for negative numbers"

    # Check backward on zero
    example_input = torch.zeros(3, 3).requires_grad_(True)
    example_output = relu(example_input)
    example_output.sum().backward()
    if example_input.grad is None:
        raise RuntimeError("Error with gradient computation in test")
    assert (example_input.grad != 0).sum() == 0, "Incorrect backward for zero"

    return None


@pytest.mark.order(3)
@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim", [(32, 5, 10), (64, 10, 1)]
)
def test_linear(batch_size: int, input_dim: int, output_dim: int) -> None:
    """
    This is the test for the Linear class.

    Args:
        batch_size: Batch size for the inputs.
        input_dim: Dimension of the input.
        output_dim: Dimension of he output.
    """

    # Define models
    model: torch.nn.Module = Linear(input_dim, output_dim)
    model_torch: torch.nn.Module = torch.nn.Linear(input_dim, output_dim)

    # Get parameters
    parameters: list[torch.nn.Parameter] = list(model.parameters())

    # Check number of parameters
    assert len(parameters) == 2, "Incorrect number of parameters, they must be 2"

    # Check shape of the weights
    assert model.weight.shape == model_torch.weight.shape, "Incorrect weight shape"

    # Check shape of the bias
    assert model.bias.shape == model.bias.shape, "Incorrect bias shape"

    # Get output
    output: torch.Tensor = model(torch.rand(batch_size, input_dim))

    # Check object
    assert isinstance(output, torch.Tensor), "Incorrect output object type"

    # Check shape of output
    assert output.shape == (batch_size, output_dim), "Incorrect shape of output"

    return None


@pytest.mark.order(4)
@pytest.mark.parametrize(
    "input_size, output_size, hidden_sizes", [(728, 10, (128, 64, 32))]
)
def test_mymodel(
    input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
) -> None:
    """
    This is the test for the MyModel class.

    Args:
        input_size: Dimension of the input.
        output_size: Dimension of the output.
        hidden_sizes: Hidden dimensions.
    """

    # Define mymodel class
    model: torch.nn.Module = MyModel(input_size, output_size, hidden_sizes)

    # Get output
    output: torch.Tensor = model(torch.rand(64, input_size))

    # Check object
    assert isinstance(output, torch.Tensor), "Incorrect output object type"

    # Check shape of output
    assert output.shape == (64, output_size), "Incorrect shape of output"

    return None
