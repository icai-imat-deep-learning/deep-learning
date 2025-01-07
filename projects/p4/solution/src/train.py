# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step
from src.utils import set_seed, save_model

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    # hyperparameters
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    hidden_size: int = 32

    # empty nohup file
    open("nohup.out", "w").close()

    # load data
    train_data: DataLoader
    val_data: DataLoader
    train_data, val_data, _, mean, std = load_data(DATA_PATH, batch_size=64)

    # define name and writer
    name: str = f"model_lr_{lr}_h_{hidden_size}_{batch_size}_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    model: torch.nn.Module = MyModel(hidden_size).to(device)

    # define loss and optimizer
    loss: torch.nn.Module = torch.nn.HuberLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(range(epochs)):
        # call train step
        train_step(model, train_data, mean, std, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, mean, std, loss, None, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
