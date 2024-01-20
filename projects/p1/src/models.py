# deep learning libraries
import torch


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        # TODO. clone inputs

        # TODO. put at zero negative elements (behavior of the ReLu)


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer. Follow the pytorch convention.

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
        """
        # call super class constructor
        super().__init__()

        # define attributes
        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(input_dim, output_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.rand(1, output_dim))

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimenions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # TODO. You will have to implement the calculation of the variable "outputs" which is a 
        # multiplication of two torch tensors (inputs and weights) plus the bias. Use the proper torch function


class MyModel(torch.nn.Module):
    """
    This is the class to construct the model. Only layers defined in
    this script can be used.
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        """
        This method is the constructor of the model.

        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """
        # call super class constructor
        super().__init__()

        # define relu
        self.relu: torch.nn.Module = ReLU()

        # TODO. Define the layers using the Linear function defined above. Take into consideration
        # the args mentioned in the header of the function

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            outputs of the model. Dimensions: [batch, 1].
        """
        # call layers
        outputs: torch.Tensor = torch.flatten(inputs, start_dim=1)
        # TODO. Implement the forward pass through the above-defined layers. 
        # Include a ReLu between each call to each layer
