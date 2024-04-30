# Final 2024

Before starting to complete the different parts, you will have to complete a small function already coded in class projects, called parameters_to_double in the src.utils module.

This part of the exam contains the following parts:

## Unfold 1D (0.5 points)

This function is contained in the src.conv1d module. Since unfold function of PyTorch is only able to work with 4D tensors (images), you will have to code this version for sequences. No nn function or for-loops are allowed except the unfold and fold function of PyTorch.

Hint: The easier way to complete this is to treat the tensors as if they are 4D matrices (images) instead of 3D (sequences).

## Fold 1D (0.5 points)
This function is contained in the src.conv1d module. Since fold function of PyTorch is only able to work with 4D tensors (images), you will have to code this version for sequences. It is needed to complete the unfold1d function before being able to pass the test. No nn function or for-loops are allowed except the unfold and fold function of PyTorch.

Hint: The easier way to complete this is to treat the tensors as if they are 4D matrices (images) instead of 3D (sequences).

## Conv1D forward (0.5 points)

This method is contained in the src.conv1d module. It is needed to complete the fold1d and unfold1d functions before. No nn functions are allowed.

## Conv1D backward (0.5 points)

This method is contained in the src.conv1d module. It is needed to complete the fold1d and unfold1d functions before. No nn functions are allowed.

## Maxout forward (1 point)

This method is contained in the src.maxout module. No nn functions are allowed.

## Maxout backward (1 point)

This method is contained in the src.maxout module. No nn functions are allowed.

## NAdam algorithm (1 point)

This class is contained in the src.optimization module. You will have to complete the constructor and step method. It is not allowed to use methods such as add or sub, only +, -, * and /. It is recommended to look at the steps of the algorithm in the PyTorch documentation.