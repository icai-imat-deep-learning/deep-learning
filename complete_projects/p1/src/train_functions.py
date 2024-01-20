# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# own modules
from src.utils import accuracy


def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the training step.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
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
        outputs: torch.Tensor = model(inputs).squeeze(1)
        loss_value: torch.Tensor = loss(outputs, targets)

        # optimize
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # add metris to lists
        losses.append(loss_value.item())
        accuracies.append(accuracy(outputs, targets).item())

    # write on tensorboard
    writer.add_scalar("train/loss", np.mean(losses), epoch)
    writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return None


def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function computes the validation step.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.
    """

    # define metric lists
    losses: list[float] = []
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
            outputs: torch.Tensor = model(inputs).squeeze(1)
            loss_value: torch.Tensor = loss(outputs, targets)

            # add metris to lists
            losses.append(loss_value.item())
            accuracies.append(accuracy(outputs, targets).item())

    # write on tensorboard
    writer.add_scalar("val/loss", np.mean(losses), epoch)
    writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)

    return None


def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.
        
    Returns:
        average accuracy.
    """

    # define metric lists
    accuracies: list[float] = []

    # activate train mode
    model.eval()

    with torch.no_grad():
        # iter over val data
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in test_data:
            # pass objects to correct device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # compute outputs and loss
            outputs: torch.Tensor = model(inputs).squeeze(1)

            # add metris to lists
            accuracies.append(accuracy(outputs, targets).item())

    accuracy_mean: float = float(np.mean(accuracies))

    return accuracy_mean
