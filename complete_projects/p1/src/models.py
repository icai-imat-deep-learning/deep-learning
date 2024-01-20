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

        # clone inputs
        outputs = inputs.clone()

        # put at zero negative elements
        outputs[outputs <= 0] = 0

        return outputs


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

        # compute outputs
        outputs: torch.Tensor = torch.matmul(inputs, self.weights) + self.bias

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
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """

        # call super class constructor
        super().__init__()

        # define relu
        self.relu: torch.nn.Module = ReLU()

        # define layers
        self.layer1: torch.nn.Module = Linear(input_size, hidden_sizes[0])
        self.layer2: torch.nn.Module = Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3: torch.nn.Module = Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4: torch.nn.Module = Linear(hidden_sizes[2], output_size)

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
        outputs = self.layer1(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer3(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer4(outputs)

        return outputs
