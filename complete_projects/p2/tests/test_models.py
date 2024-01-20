# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.models import Block


@pytest.mark.order(1)
@pytest.mark.parametrize(
    "input_channels, output_channels, stride", [(3, 10, 2), (1, 20, 4)]
)
def test_block(input_channels: int, output_channels: int, stride: int) -> None:
    """
    This is a test for the Block class.

    Args:
        input_channels: inut channels.
        output_channels: output channels.
        stride: stride.
    """

    # define block
    block: torch.nn.Module = Block(input_channels, output_channels, stride)

    # check sequential object
    assert isinstance(
        list(block.children())[0], torch.nn.Sequential
    ), "Layers not encapsulated inside sequential"

    # check length
    assert len(list(block.children())[0]) == 6, "Incorrect length "
    
    if isinstance(list(block.children())[0], torch.nn.Sequential):
        sequential: torch.nn.Sequential = list(block.children())[0]
        
        # check stride 1 in first conv layer
        assert list(block.children())[0][0].stride == (
            1,
            1,
        ), "Incorrect stride in first conv layer"

        # check stride 1 in first conv layer
        assert list(block.children())[0][2].stride == (
            2,
            2,
        ), "Incorrect stride in first conv layer"

    # define input and compute output
    example_input: torch.Tensor = torch.rand(64, input_channels, 224, 224)
    example_output: torch.Tensor = block(example_input)

    # check type of object
    assert isinstance(example_output, torch.Tensor), "Incorrect output object type"

    # check dimensions
    assert example_output.shape == (
        64,
        output_channels,
        224 // stride,
        224 // stride,
    ), "Incorrect shape of output object"

    return None
