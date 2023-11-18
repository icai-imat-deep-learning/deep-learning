# deep learning libraries
import torch


class MyModel(torch.nn.Module):
    """
    This is the class to construct the model
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple[int, int, int]
    ) -> None:
        """
        This method is the constructor of the model

        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_sizes: three hidden sizes of the model
        """

        # define norm layer
        self.norm_layer: torch.nn.Module = torch.nn.BatchNorm1d(input_size)

        # define relu
        self.relu: torch.nn.Module = torch.nn.ReLU()

        # define layers
        self.layer1: torch.nn.Module = torch.nn.Linear(input_size, hidden_sizes[0])
        self.layer2: torch.nn.Module = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3: torch.nn.Module = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4: torch.nn.Module = torch.nn.Linear(hidden_sizes[2], output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model

        Args:
            inputs: input tensor, Dimensions: [batch, features]

        Returns:
            outputs of the model. Dimensions: [batch, 1]
        """

        # call layers
        outputs: torch.Tensor = self.norm(inputs)
        outputs = self.layer1(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer3(outputs)
        outputs = self.relu(outputs)
        outputs = self.layer4(outputs)

        return outputs
