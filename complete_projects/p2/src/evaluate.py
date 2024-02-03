# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
import os
from tqdm.auto import tqdm
from typing import Dict, Union, Literal

# own modules
from src.models import CNNModel
from src.utils import (
    load_imagenette_data,
    Accuracy,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 10


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
    ) = load_imagenette_data(DATA_PATH, batch_size=128, num_workers=4)

    # define model
    model = CNNModel(output_channels=NUMBER_OF_CLASSES).to(device)

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
    print(f"accuracy: {main('best_model.pt')}")
