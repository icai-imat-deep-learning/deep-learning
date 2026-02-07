"""
This module contains the code to evaluate the models.
"""

# Standard libraries
from typing import Final

# 3pps
import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader

# Own modules
from src.train_functions import t_step
from src.utils import load_data

# Static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program
    """

    # Load data
    test_data: DataLoader
    _, _, test_data = load_data(DATA_PATH, batch_size=64)

    # Define name and writer
    name: str = "best_model"

    # Define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt").to(device)

    # Call test step and evaluate accuracy
    accuracy: float = t_step(model, test_data, device)
    print(f"accuracy: {accuracy}")

    return None


if __name__ == "__main__":
    main()
