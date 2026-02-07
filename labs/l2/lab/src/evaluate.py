"""
This module contains rhe data for the evaluation of the models.
"""

# 3pps
import torch
from torch.jit import RecursiveScriptModule

# Own modules
from src.utils import (
    load_imagenette_data,
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

# Static variables
DATA_PATH: str = "data"


def main(name: str) -> float:
    """
    This function is the main program for the testing.
    """

    # TODO


if __name__ == "__main__":
    print(f"accuracy: {main('best_model')}")
