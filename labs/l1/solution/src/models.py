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

        # Call super class constructor
        super().__init__()
        
        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
        """

        # Clone inputs
        outputs = inputs.clone()

        # Put at zero negative elements
        outputs[outputs <= 0] = 0

        return outputs


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
        PyTorch convention.

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
        """

        # Call super class constructor
        super().__init__()

        # Define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.rand(input_dim, output_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.rand(1, output_dim))

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
        """

        # Compute outputs
        outputs: torch.Tensor = torch.matmul(inputs, self.weight) + self.bias

        return outputs


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

        # Call super class constructor
        super().__init__()

        # Define relu
        self.relu: torch.nn.Module = ReLU()

        # Define layers
        self.layer1: torch.nn.Module = Linear(input_size, hidden_sizes[0])
        self.layer2: torch.nn.Module = Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3: torch.nn.Module = Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4: torch.nn.Module = Linear(hidden_sizes[2], output_size)
        
        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            Inputs: input tensor, Dimensions: [batch, channels, height,
                width].

        Returns:
            Outputs of the model. Dimensions: [batch, 1].
        """

        # call layers
        outputs: torch.Tensor = torch.flatten(inputs, start_dim=1)
        outputs = self.layer1(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer3(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer4(outputs)

        return outputs
