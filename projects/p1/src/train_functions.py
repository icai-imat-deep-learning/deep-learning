# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    # define metric lists
    losses: list[float]
    accuracies: list[float] = []

    # activate train mode
    model.train()

    # iter over training data
    inputs: torch.Tensor
    targets: torch.Tensor
    for inputs, targets in train_data:
        # pass objects to correct device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute outputs and loss
        outputs: torch.Tensor = model(inputs).unsqueeze(1)
        loss_value: torch.Tensor = loss(outputs, targets)

        # optimize
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # add metris to lists
        losses.append(loss_value.item())
        accuracies.append(loss_value.item())

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)

    return None


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    # define metric lists
    losses: list[float]
    accuracies: list[float] = []

    # activate train mode
    model.eval()

    with torch.no_grad():
        # iter over val data
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in val_data:
            # pass objects to correct device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # compute outputs and loss
            outputs: torch.Tensor = model(inputs).unsqueeze(1)
            loss_value: torch.Tensor = loss(outputs, targets)

            # add metris to lists
            losses.append(loss_value.item())
            accuracies.append(loss_value.item())

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)

    return None
