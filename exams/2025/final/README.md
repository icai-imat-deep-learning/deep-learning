# Final 2025

This is the final exam of 2025. For this exam you are only allowed to use pytorch documentation and pytorch forums. In order to access those you can use a browser as google but only access pytorch documentation and pytorch forums pages. Other web pages, generative AI tools or any other material as previous projects are forbidden.

Moreover, for this exam you cannot use any class or function from torch.nn module, and operations must be done with indexing (e.g. torch.where is not allowed).

## MaxOut (5 points)

You have to code MaxOut layer in pytorch for a generic number of units. For this implementation you cannot use loops, so everything has to be vectorized operations. You have the formula and diagram of the MaxOut in the artifacts folder. Therefore, we will use the [batch matrix multiplication](https://pytorch.org/docs/stable/generated/torch.bmm.html) (BMM). We advice to have a look at it, since understanding the formula and shapes is a key step to complete the exercise. In this case you have to complete the following functions:

### init method (0.5 points)

Here you will have to define the constructor and the objects that will be used.

### reshape_inputs (1 point)

This method reshapes the inputs in a way they can be used in the BMM. For this function the target shape is provided in the docstring of the function and also a test is available. Take account that for the reshaping you may have to use repetitions and other things, and not only view operations.

### reshape_weight (2 points)

This method reshapes the weight tensor created in the init method in a way that it can be used in the BMM. For this function the target shape is not provided and there is not available test, you will have to figure out if the reshaping is okey in the test for the forward pass. Take account that for the reshaping you may have to use repetitions and other things, and not only view operations.

### forward (1.5 points)

This method is the complete forward pass. You will have to use the two previous functions and the BMM to perform the MaxOut. 