# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.utils import set_seed
from src.models import Dropout


def test_dropout() -> None:
    # define inputs
    original_inputs: torch.Tensor = torch.rand(64, 30)
    inputs: torch.Tensor = original_inputs.clone()

    # define dropout
    dropout = Dropout(0.5)
    dropout_torch: torch.nn.Module = torch.nn.Dropout(0.5)

    for mode in ("train", "eval"):
        # activate mode
        if mode == "train":
            # activate train mode
            dropout.train()
            dropout_torch.train()

        else:
            # activate eval mode
            dropout.eval()
            dropout_torch.eval()

        # compute outputs
        set_seed(42)
        outputs: torch.Tensor = dropout(inputs)
        set_seed(42)
        outputs_torch: torch.Tensor = dropout_torch(inputs)

        # check outputs of dropout
        assert (outputs != outputs_torch).sum().item() == 0, (
            f"Incorrect outputs when {mode} mode activated, outputs are not equal to "
            f"pytorch implementation"
        )
