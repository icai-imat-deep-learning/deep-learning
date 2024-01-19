# Project 1

This project will be graded in the following way, 7 points will be graded by automtic tests, that can be verified by the criteria of a professor if it is neccesary. The remaining 3 points will come from the inspection of the code by a professor.

The goal of this first lab is to implement the required source code to train and test a neural network to recognize cyphers based on the common dataset denominated MNIST (https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

This dataset consists on a large set of handwritten digits. The provided source code is a pytorch project with the function to load data, train the data, test it and extract metrics. Parts of the source code is missing and your will have to complete it to be able to successfully train the network.

As commented below, we recommend you to implement the source code sequentially step by step and its corresponding test (QA or quality assurance) to evidence the correct behaviour of each part that you develop. 

The first thing you will have to do is installing the required dependencies, defined in the file "requirements.txt".

## Structure of the repo



## Automatic tests (QA)
These tests are already provided to you, you can run them by executing 

    pytest .

At the beginning these test will fail and you will have to implement the required functions properly for them to be set as PASS.


## Inspection of the code

These 3 points are meant to assess things that cannot be measured by automatic testing, as code tyle and organization.


## Parts of the project

We recommend the student to follow the order we present in this section since it is the easiest and mosre natural one to complete the project.

### load_data function (QA test, 1 point)

This function is contained in the src.utils module and must be completed to load the three dataloaders of train, val and test in respective order. In order to do that, the division between train and val must be 0.8-0.2. Finally, all batches should be equal size.  

To execute the unit test implemented to assure the correct implementation of the load_data function, run:
pytest tests/test_utils.py::test_load_data

This will be analog for the rest of the functions.

### ReLU class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations. The functionality must be exactly the same as the torch.nn.ReLU class from PyTorch.

### Linear class (QA test, 1 point)

This class is contained in the src.models and must be completed using only matrix operations. The conventions used must be the PyTorch ones.

### MyModel class (QA test, 1 point)

This class is contained in the src.models and must be completed by calling the Linear and ReLU classes and pytorch operations (without PyTorch NN layers).

### Train step (Inspection)

This is the training step for each epoch. It should train the paremeters, compute the average loss and accuracy and log them into tensorboard.

### Validation step (Inspection)

This is the validation step for each epoch. It should compute the average loss and accuracy and log them into tensorboard.

### Testing step (Inspection)

This is the test step for each epoch. It should compute the average accuracy and return it.

### accuracy function (1 point)

This function is contained in the src.utils module and must be completed to compute the accuracy of the datasets.

### Performance (2 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 94% would have a score of 1 point, higher than 96% 1.5 and higher than 97% 2 points. You should try to play with the hyperparameters and then rename you best model as best_model.pt inside the models folder. Please note that the test implemented for the performance part will fail if your train model does not reach the accuracy thresholds defined above.

