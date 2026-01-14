"""
This module contains the tests for the src.utils module.
"""

# 3pps
import torch
from torch.utils.data import DataLoader
import pytest

# Own modules
from src.utils import load_data, accuracy, set_seed


@pytest.mark.order(1)
@pytest.mark.parametrize("data_path", ["data"])
def test_load_data(data_path: str) -> None:
    """
    This is the test for the load_data function.

    Args:
        data_path: path of the data.
    """
    
    # Get datasets
    datasets: tuple[DataLoader, DataLoader, DataLoader] = load_data(
        data_path, batch_size=64
    )

    # Check length of dataloaders
    assert len(datasets) == 3, "Incorrect length of dataloaders, it should be three"

    # Check train length
    assert len(datasets[0]) == 750, "Incorrect length of training dataset"

    # Check validation length
    assert len(datasets[1]) == 187, "Incorrect length of validation dataset"

    # Check test length
    assert len(datasets[2]) == 156, "Incorrect length of test dataset"

    # Get element
    batch: tuple[torch.Tensor, torch.Tensor] = next(iter(datasets[0]))

    # Check batch images shape
    assert batch[0].shape == (64, 1, 28, 28), "Incorrect size of batch images"

    # Check batch images shape
    assert batch[1].shape == (64,), "Incorrect size of batch images"

    # Check drop last arguments
    assert datasets[0].drop_last, "Not every batch is equal size"

    return None


@pytest.mark.order(5)
def test_accuracy() -> None:
    """
    This is the test for teh accuracy function.
    """

    # Set seed
    set_seed(42)

    # define predictions and target
    predictions: torch.Tensor = torch.rand(64, 10)
    targets: torch.Tensor = torch.argmax(predictions, dim=1)
    targets[5:10] = 0

    # calculate accuracy value
    accuracy_value: float = float(accuracy(predictions, targets))

    # check accuracy number
    assert (
        accuracy_value == 0.921875
    ), f"Incorrect accuracy number, it should be 0.921875 instead of {accuracy_value}"

    return None
