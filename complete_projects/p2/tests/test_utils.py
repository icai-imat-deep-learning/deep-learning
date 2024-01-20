# deep learning libraries
import torch
from torch.utils.data import Dataset

# other libraries
import os
import pytest

# own modules
from src.utils import download_data, ImagenetteDataset


@pytest.mark.order(1)
@pytest.mark.parametrize("path", ["data"])
def test_imagenette(path: str) -> None:
    """
    This fucntion is the test for Imagenette class.

    Args:
        path: path for saving the data
    """

    # download folders if they are not present
    if not os.path.isdir(f"{path}"):
        # create main dir
        os.makedirs(f"{path}")

        # download data
        download_data(path)

    # create datasets
    train_dataset: ImagenetteDataset = ImagenetteDataset(f"{path}/train")
    test_dataset: ImagenetteDataset = ImagenetteDataset(f"{path}/val")

    # check train length
    assert (
        len(train_dataset) == 9296
    ), f"Incorrect length, got {len(train_dataset)} and it should be 9296"

    # check test length
    assert (
        len(test_dataset) == 3856
    ), f"Incorrect length, got {len(test_dataset)} and it should be 3856"

    # get example of output
    element: tuple[torch.Tensor, int] = train_dataset[0]

    # check number of objects returned by __getitem__
    assert len(element) == 2, (
        f"Incorrect number of objects returned by __getitem__ method, 2 were expected and got "
        f"{len(element)}"
    )

    # check first object type
    assert isinstance(
        element[0], torch.Tensor
    ), "Incorrect object type of first element of __getitem__ output"

    # check first object type
    assert isinstance(
        element[1], int
    ), "Incorrect object type of second element of __getitem__ output"

    # check first object
    assert element[0].shape == (3, 224, 224), "Incorrect shape of image tensor"

    return None
