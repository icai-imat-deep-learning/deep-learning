# Project 1

This project will be graded in the following way, 7 points will be graded by automatic tests, that can be verified by the criteria of a professor if it is necessary. The remaining 3 points will come from the inspection of the code by a professor.

# Structure of the repo
In this repo, all the functional code is inside the src/ folder. For training the model and save it, the following module must be run:

    python -m src.train

In the src.train module there are some hyperparameters at the beginning of the main function that can be modified to optimize the model. For evaluating the model the following module can be run:

    python -m src.evaluate

The evaluate file will load a model called best_model.pt inside the models/ folder compute accuracy in the test set and print it in the command line. 


# Automatic tests (QA)
These tests are already provided to you, you can run them by executing:

    pytest .

The grades achieved by these tests can be overridden by the grade from a professor if the test is fulfilled but the goal of the function/class to be completed is not reached.

# Inspection of the code

These 3 points are meant to assess things that cannot be measured by automatic testing, such as code style and organization.


# Parts of the project

We recommend the student follow the order we present in this section since it is the easiest and most natural one to complete the project.

### load_data function (QA test, 1 point)

This function is contained in the src.utils module and must be completed to load the three dataloaders of train, val and test in their respective order. To do that, the division between train and val must be 0.8-0.2. Finally, all batches should be equal in size.   

### ReLU class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations. The functionality must be the same as the torch.nn.ReLU class from PyTorch.

### Linear class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations. The conventions used must be the PyTorch ones.

### MyModel class (QA test, 1 point)

This class is contained in the src.models and must be completed by calling the Linear and ReLU classes and PyTorch operations (without PyTorch NN layers).

### Train step (Inspection)

This is the training step for each epoch. It should train the parameters, compute the average loss and accuracy and log them into tensorboard.

### Validation step (Inspection)

This is the validation step for each epoch. It should compute the average loss and accuracy and log them into tensorboard.

### Testing step (Inspection)

This is the test step for each epoch. It should compute the average accuracy and return it.

### accuracy function (1 point)

This function is contained in the src.utils module and must be completed to compute the accuracy of the datasets.

### Performance (2 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 94% would have a score of 1 point, higher than 96% 1.5 and higher than 97% 2 points. You should try to play with the hyperparameters and then rename your best model as best_model.pt inside the models' folder.

