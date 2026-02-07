"""
This module contains the code to train models.
"""

# Standard libraries
from typing import Final

# 3pps
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Own modules
from src.models import MyModel
from src.train_functions import train_loop, val_loop
from src.utils import load_data, save_model

# Static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program for training.
    """

    # Hyperparameters
    epochs: int = 2
    lr: float = 1e-3
    batch_size: int = 64
    hidden_sizes: tuple[int, ...] = (256, 128, 64)

    # Empty nohup file
    open("nohup.out", "w", encoding="utf-8").close()

    # Load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _ = load_data(DATA_PATH, batch_size=64)

    # Define name and writer
    name: str = f"model_lr_{lr}_hs_{hidden_sizes}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # Define model
    inputs: torch.Tensor = next(iter(train_data))[0]
    model: torch.nn.Module = MyModel(
        inputs.shape[2] * inputs.shape[3], NUM_CLASSES, hidden_sizes=hidden_sizes
    ).to(device)

    # Define loss and optimizer
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in tqdm(range(epochs)):
        # Call train step
        train_loop(model, train_data, loss, optimizer, writer, epoch, device)

        # Call val step
        val_loop(model, val_data, loss, writer, epoch, device)

    # Save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
