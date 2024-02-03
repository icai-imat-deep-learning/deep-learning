# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
import os
from tqdm.auto import tqdm
from typing import Dict, Union, Literal

# own modules
from src.models import CNNModel
from src.utils import (
    load_imagenette_data,
    Accuracy,
    set_seed,
)

# set device
device: torch.device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# set all seeds and set number of threads
set_seed(42)
torch.set_num_threads(8)

# static variables
DATA_PATH: str = "data"

NUMBER_OF_CLASSES: int = 10


def main() -> None:
    """
    This function is the main program for the training.
    """

    # hyperparameters
    lr: float = 1e-4
    epochs: int = 100

    # empty nohup file
    open("nohup.out", "wb").close()

    # check device
    print(f"device: {device}")

    train_data, val_data, _ = load_imagenette_data(
        DATA_PATH, batch_size=128, num_workers=4
    )

    # define model name and tensorboard writer
    name = f"model_lr_{lr}_e_{epochs}"
    writer = SummaryWriter(f"runs/{name}")

    # define model
    model = CNNModel((32, 64, 128), output_channels=NUMBER_OF_CLASSES).to(device)

    # define loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # define accuracy
    accuracy: Accuracy = Accuracy()

    # define progress bar
    progress_bar = tqdm(range(epochs * (len(train_data) + len(val_data))))

    # epochs loop
    for epoch in range(epochs):
        # train mode
        model.train()

        # reset accuracy and initialize losses list
        accuracy.reset()
        losses = []

        # train step loop
        for images, labels in train_data:
            # pass images and labels to the correct device
            images = images.to(device)
            labels = labels.to(device)

            # compute outputs and loss
            outputs = model(images)
            loss_value = loss(outputs, labels.long())

            # compute gradient and update parameters
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # add metrics to vectors
            losses.append(loss_value.item())
            accuracy.update(outputs, labels)

            # progress bar step
            progress_bar.update()

        # write results on tensorboard
        writer.add_scalar("loss/train", np.mean(losses), epoch)
        writer.add_scalar("accuracy/train", accuracy.compute(), epoch)

        # evaluation mode
        model.eval()
        with torch.no_grad():
            # reset accuracy
            accuracy.reset()

            # val step loop
            for images, labels in val_data:
                # pass images and labels to the correct device
                images = images.to(device)
                labels = labels.to(device)

                # compute outputs and loss
                outputs = model(images)
                loss_value = loss(outputs, labels.long())

                # add metrics to vectors
                losses.append(loss_value.item())
                accuracy.update(outputs, labels)

                # progress bar step
                progress_bar.update()

            # write results on tensorboard
            writer.add_scalar("accuracy/val", accuracy.compute(), epoch)

    # save model
    if not os.path.exists(f"models"):
        os.makedirs(f"models")
    torch.save(model, f"models/{name}.pt")


if __name__ == "__main__":
    main()
