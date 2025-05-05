"""
This module contains the code for MaxOut.
"""

# 3pps
import torch


class MaxOut(torch.nn.Module):
    """
    This class implements the MaxOut layer without loops.

    Attr:
        num_units: Number of linear layers the MaxOut s going to use.
        weight: Tensor object with all the weights of the different
            layers. Dimensions: [num_units, output dim, input dim].
            The dtype is a double.
    """

    # Define attributes
    num_units: int
    weight: torch.Tensor

    def __init__(self, num_units: int, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the class.
        
        Returns:
            None.
        """

        # Call super class
        super().__init__()

        # TODO

    def reshape_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method reshapes the inputs in a way later can be used to perform
        matrix multiplication.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Inputs reshaped. Dimensions: [number of units * batch size,
                1, input dim].
        """

        # TODO

    def reshape_weight(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This function transform the dimensions of the weight so it can
        be multiplied with bmm.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Weights reshaped.
        """

        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch size, input dim].

        Returns:
            Output tensor. Dimensions: [batch size, output dim].
        """

        # TODO
