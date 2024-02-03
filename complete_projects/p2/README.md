# Project 2

This project will be graded in the following way, 8 points will be graded by automatic tests, that can be verified by the criteria of a professor if it is necessary. The remaining 2 points will come from the inspection of the code by a professor. https://web.eecs.umich.edu/~justincj/teaching/eecs442/notes/linear-backprop.html

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

### Imagenette class (2 points)

This class is contained in the src.utils module. This class is the dataset to load images from the memory and then, combined with the dataloader, creates the batches to train the model. 

### Block class (1 point)

This class is contained in the src.models module. 


### 

### Mypy type checking (1.5 points)

The code must be properly typed hinted and pass the mypy type checker. To run the checker, the following command must be executed:

    mypy --cache-dir=/dev/null --check-untyped-defs --ignore-missing-imports .

It is also available in the pre_commit.sh file, that can be executed as follows:

    source pre_commit.sh

The benefit of this second file is that by running the pre_commit your code will also be formatted.

### Black format (0.5 points)

The code must be formatted following the black style. In order to achieve that, it 

### Performance (3 points)

This is not a specific function but a performance you should achieve with your best model in the test set. A performance higher than 94% would have a score of 1 point, higher than 96% 1.5 and higher than 97% 2 points. You should try to play with the hyperparameters and then rename your best model as best_model.pt inside the models' folder.

