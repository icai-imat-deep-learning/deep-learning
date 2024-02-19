# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed
from src.models import Dropout


@pytest.mark.order(6)
def test_dropout() -> None:
    # define inputs
    inputs: torch.Tensor = torch.rand(64, 30)
    inputs_torch: torch.Tensor = inputs.clone()

    # define dropout
    dropout = Dropout(0.5)
    dropout_torch: torch.nn.Module = torch.nn.Dropout(0.5)

    # activate train mode
    dropout.train()
    dropout_torch.train()

    # compute outputs
    set_seed(42)
    outputs: torch.Tensor = dropout(inputs)
    set_seed(42)
    outputs_torch: torch.Tensor = dropout_torch(inputs)

    # check output type
    assert isinstance(
        outputs, torch.Tensor
    ), f"Incorrect type, expected torch.Tensor got {type(outputs)}"

    # check output size
    assert (
        outputs.shape == inputs.shape
    ), f"Incorrect shape, expected {inputs.shape}, got {outputs.shape}"

    # check outputs of dropout
    assert (outputs != outputs_torch).sum().item() == 0, (
        "Incorrect outputs when train mode activated, outputs are not equal to "
        "pytorch implementation"
    )

    # activate eval mode
    dropout.eval()
    dropout_torch.eval()

    # compute outputs
    set_seed(42)
    outputs = dropout(inputs)
    set_seed(42)
    outputs_torch = dropout_torch(inputs)

    # check outputs of dropout
    assert (outputs != outputs_torch).sum().item() == 0, (
        "Incorrect outputs when eval mode activated, outputs are not equal to "
        "pytorch implementation"
    )

    # define dropout with inplace
    dropout = Dropout(0.5, inplace=True)
    dropout_torch = torch.nn.Dropout(0.5, inplace=True)

    # compute outputs
    set_seed(42)
    dropout(inputs)
    set_seed(42)
    dropout_torch(inputs_torch)

    # check outputs of dropout
    assert (inputs != inputs_torch).sum().item() == 0, (
        "Incorrect outputs when inplace is activated, outputs are not equal to "
        "pytorch implementation"
    )

    return None
