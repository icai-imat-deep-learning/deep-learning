# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm

# own modules
from src.utils import load_data, save_model
from src.models import MyModel
from src.train_functions import train_step, val_step

# static variables
DATA_PATH = "data/train.csv"

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main() -> None:
    """
    This function is the main program
    """

    # hyperparameters
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64

    # load data
    train_data: DataLoader
    val_data: DataLoader
    counts: tuple[float, float]
    train_data, val_data, counts = load_data(DATA_PATH, 64)

    # define name and writer
    name: str = f"e_{epochs}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define model
    inputs: torch.Tensor = next(iter(train_data))
    model: torch.nn.Module = MyModel(inputs.shape[0], 1, (256, 128, 64)).to(device)

    # define loss and optimizer
    pos_weight: torch.Tensor = torch.empty(batch_size)
    loss: torch.nn.Module = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([counts[0] / counts[1]])
    )
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train loop
    for epoch in tqdm(epochs):
        # call train step
        train_step(model, train_data, loss, optimizer, writer, epoch, device)

        # call val step
        val_step(model, val_data, loss, writer, epoch, device)

    # save model
    save_model(model, name)

    return None


if __name__ == "__main__":
    main()
