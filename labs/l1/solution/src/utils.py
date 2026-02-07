"""
This module contains auxiliary functionality.
"""

# Standard libraries
import os
import random

# 3pps
import numpy as np
import torch
import torchvision
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def load_data(
    path: str, batch_size: int = 128
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    This function loads the data from MNIST dataset, using the
    torchvision function (URL provided in the README). All batches must
    be equal size. The division between train and val must be 0.8-0.2.

    Args:
        path: path to save the datasets (train and test).
        batch_size: batch size. Defaults to 128.

    Returns:
        tuple of three dataloaders, train, val and test in respective order.
    """

    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()])

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=transformations
    )
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2])
    test_dataset = torchvision.datasets.MNIST(
        root=path, train=False, download=True, transform=transformations
    )

    # Define dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # Create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: Seed number to fix radomness.
    """

    # Set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # Set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
        predictions: Predictions tensor. Dimensions:
            [batch, num classes] or [batch].
        targets: Targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        Accuracy in a tensor of a single element.
    """

    # Eliminate extra dimension
    if len(targets.shape) > 1:
        targets.squeeze(1)

    # Compute predictions
    predictions = torch.argmax(predictions, dim=1)

    # Compute accuracy
    score: torch.Tensor = (predictions == targets).sum() / predictions.shape[0]

    return score
