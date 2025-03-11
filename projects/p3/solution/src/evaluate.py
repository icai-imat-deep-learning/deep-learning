"""
This module contains the code to evaluate the models.
"""

# 3pps
import torch
from torch.jit import RecursiveScriptModule

# Own modules
from src.data import load_data
from src.utils import (
    Accuracy,
    load_model,
    set_seed,
)

# Set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# Set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # check device
    print(f"device: {device}")

    (
        _,
        _,
        test_data,
    ) = load_data(DATA_PATH, batch_size=128, num_workers=4)

    # define model
    model: RecursiveScriptModule = load_model(name).to(device)

    # define accuracy
    accuracy: Accuracy = Accuracy()

    # train step loop
    for images, labels in test_data:
        # pass images and labels to the correct device
        images = images.to(device)
        labels = labels.to(device)

        # compute outputs and loss
        outputs = model(images)
        accuracy.update(outputs, labels)

    return accuracy.compute()


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
