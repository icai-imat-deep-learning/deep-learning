"""
This module contains the code for the models.
"""

# 3pps
import torch


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self) -> None:
        """
        This method is the constructor of the ReLU layer.
        """

        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
        """

        # TODO


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight for the linear transformation.
        Bias: Bias term for the linear transformation.
    """

    # Define attributes
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        PyTorch convention. For the shapes you must look in the
        PyTorch Linear layer documentation.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
        """

        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
        """

        # TODO


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
            input_size: Size of the input
            output_size: Size of the output
            hidden_sizes: Three hidden sizes of the model
        """

        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            Inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            Outputs of the model. Dimensions: [batch, 1].
        """

        # TODO
