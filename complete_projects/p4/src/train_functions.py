# deep learning libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from typing import Optional


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        optimizer: optimizer.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # define mae and mse functions
    mae: torch.nn.Module = torch.nn.L1Loss()
    mse: torch.nn.Module = torch.nn.MSELoss()

    # initialize loss vectors
    mae_vector: list[float] = []
    mse_vector: list[float] = []
    losses: list[float] = []

    # train
    model.train()

    # train loop
    inputs: torch.Tensor
    targets: torch.Tensor
    for inputs, targets in train_data:
        # prepare data
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)

        # compute output and loss
        outputs: torch.Tensor = model(inputs)[:, -1, :]
        loss_value: torch.Tensor = loss(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # add loss and mae for the step
        losses.append(loss_value.item())
        mae_vector.append(
            mae(
                outputs * std + mean,
                targets * std + mean,
            ).item()
        )
        mse_vector.append(
            mse(
                outputs * std + mean,
                targets * std + mean,
            ).item()
        )

    # add measures to tensorboard
    writer.add_scalar("loss", np.mean(losses), epoch)
    writer.add_scalar("mae/train", np.mean(mae_vector), epoch)
    writer.add_scalar("mse/train", np.mean(mse_vector), epoch)
    if optimizer.param_groups[0]["lr"] is not None:
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    return None


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    mean: float,
    std: float,
    loss: torch.nn.Module,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    writer: SummaryWriter,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        mean: mean of the target.
        std: std of the target.
        loss: loss function.
        scheduler: scheduler.
        writer: writer for tensorboard.
        epoch: epoch of the training.
        device: device for running operations.
    """

    # define mae and mse functions
    mae: torch.nn.Module = torch.nn.L1Loss()
    mse: torch.nn.Module = torch.nn.MSELoss()

    # initialize loss vectors
    mae_vector: list[float] = []
    mse_vector: list[float] = []

    # activate eval mode
    model.eval()

    # evaluate
    with torch.no_grad():
        # val loop
        for inputs, targets in val_data:
            # prepare data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            # compute output and loss
            outputs = model(inputs)[:, -1, :]

            # add loss and mae for the step
            mae_vector.append(
                mae(
                    outputs * std + mean,
                    targets * std + mean,
                ).item()
            )
            mse_vector.append(
                mse(
                    outputs * std + mean,
                    targets * std + mean,
                ).item()
            )

        # add measures to tensorboard
        writer.add_scalar("mae/val", np.mean(mae_vector), epoch)
        writer.add_scalar("mse/val", np.mean(mse_vector), epoch)

    # update scheduler
    if scheduler is not None:
        scheduler.step()

    return None


@torch.no_grad()
def t_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    mean: float,
    std: float,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        mean: mean of the target.
        std: std of the target.
        device: device for running operations.

    Returns:
        mae of the test data.
    """

    # define mae and mse functions
    mae: torch.nn.Module = torch.nn.L1Loss()

    # activate eval mode
    model.eval()

    # evaluate
    with torch.no_grad():
        # initialize losses vectors
        mae_vector: list[float] = []

        # test loop
        inputs: torch.Tensor
        targets: torch.Tensor
        for inputs, targets in test_data:
            # prepare data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)

            # compute output and loss
            outputs: torch.Tensor = model(inputs)[:, -1, :]

            # add loss and mae for the step
            mae_vector.append(
                mae(
                    outputs * std + mean,
                    targets * std + mean,
                ).item()
            )

        # add measures to tensorboard
        mae_value: float = float(np.mean(mae_vector))

    return mae_value
