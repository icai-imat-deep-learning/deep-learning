# deep learning libraries
import torch
import numpy as np
import pandas as pd
from torch.jit import RecursiveScriptModule
from torch.utils.data import Dataset, DataLoader, random_split

# other libraries
import os
import random


def load_data(
    path: str, batch_size: int
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    # load raw data
    X: np.ndarray
    y: np.ndarray
    X, y = load_raw_data(path)

    counts_array: np.ndarray
    _, counts_array = np.unique(y, return_counts=True)
    counts_array = counts_array * 100 / len(y)
    counts: tuple[float, float] = (counts[0], counts[1])

    # create dataset
    dataset: Dataset = MyDataset(X, y)

    # split into train and val
    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    # define dataloaders
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    return train_dataloader, val_dataloader, counts


def load_raw_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    This function loads the raw data from a csv

    Args:
        path: path of the csv file

    Returns:
        tuple of numpy arrays, the inputs and the targets respectively
    """

    df = pd.read_csv(path)
    y = df["TARGET"].to_numpy()
    df = df.drop(["ID", "TARGET"], axis=1)
    X = df.to_numpy()

    return X, y


class MyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        inputs: torch.Tensor = torch.from_numpy(self.X[index])
        target: int = self.y[index]

        return inputs, target


def save_model(model, name) -> None:
    if os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Args:
        seed: seed number to fix radomness
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
