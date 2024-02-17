# deep learning libraries
import torch
from torch.optim.lr_scheduler import LRScheduler

# other libraries
import pytest

# own modules
from src.utils import StepLR, set_seed

# set seed
set_seed(42)


@pytest.mark.order(1)
def test_steplr() -> None:
    # define model
    model: torch.nn.Module = torch.nn.Sequential(
        torch.nn.Linear(30, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )

    # define inputs and targets
    inputs: torch.Tensor = torch.rand(64, 30)
    targets: torch.Tensor = torch.rand(64)

    # define loss and lr
    loss: torch.nn.Module = torch.nn.L1Loss()
    lr: float = 1e-3

    # define optimizers
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_torch: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # define schedulers
    scheduler: LRScheduler = StepLR(optimizer, step_size=50, gamma=0.2)
    scheduler_torch: LRScheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_torch, 50, gamma=0.2
    )

    # iter over epochs loop
    for epoch in range(110):
        # compute steps
        scheduler.step()
        scheduler_torch.step()

        # get lr and compare them
        lr = optimizer.param_groups[0]["lr"]
        lr_torch: float = optimizer_torch.param_groups[0]["lr"]
        print(lr)
        assert (
            lr == lr_torch
        ), f"Incorrect step of scheduler, expected {lr_torch} in {epoch} epoch, and got {lr}"

    return None


# @pytest.mark.order(1)
# def test_adam() -> None:

#     # define model
#     model_original: torch.nn.Module = torch.nn.Sequential(
#         torch.nn.Linear(30, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )

#     # clone model
#     model1: torch.nn.Module = model_original.clone()
#     model2: torch.nn.Module = model_original.clone()

#     # define inputs and targets
#     inputs: torch.Tensor = torch.rand(64, 30)
#     targets: torch.Tensor = torch.rand(64)

#     # define loss and lr
#     loss = torch.nn.L1Loss()
#     lr: float = 1e-3

#     # define optimizers
#     optimizer1: torch.optim.Optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
#     optimizer2: torch.optim.Optimizer = ADAMOptimizer(model2.parameters(), lr=lr)

#     # optimize first model
#     outputs: torch.Tensor = model1(inputs)
#     loss_value: torch.Tensor = loss(outputs, targets)
#     optimizer1.zero_grad()
#     loss_value.backward()
#     optimizer1.step()

#     # optimize second model
#     outputs = model2(inputs)
#     loss_value = loss(outputs, targets)
#     optimizer2.zero_grad()
#     loss_value.backward()
#     optimizer2.step()

#     assert model1.parameters() == model2.parameters(), "Incorrect return of the algorithm"
