# deep learning libraries
import torch

# import torchvision


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of the Block class. It is composed of
        3x(conv(kernel=3) + ReLU). Only the second conv
        will have stride. Use a Sequential for encapsulating all the
        layers. Clue: convs may have padding to fit into the correct
        dimensions.

        Args:
            input_channels: input channels for Block.
            output_channels: output channels for Block.
            stride: stride only for the second convolution of the
                Block.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # fill network
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the foward pass.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, input_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        return self.net(inputs)


class CNNModel(torch.nn.Module):
    """
    Neural net composed of conv (kernel 7), ReLU, max pool (kernel 3),
    Block and linear. Note: in the forward method, before classifier
    layer a GAP is performed.

    Attributes:
        network (torch.nn.Module): neural net composed of conv layers,
            ReLUs and a max pooling.
        classifier (torch.nn.Module): a linear layer.
    """

    def __init__(
        self,
        layers: tuple[int, int, int] = (32, 64, 128),
        input_channels: int = 3,
        output_channels: int = 10,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the SuperTuxBlocks.
            input_channels: input channels of the model.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # initialize module_list with a conv of kernel 7 a ReLU and a max pooling of kernel 3
        module_list = [
            torch.nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        # add 3 Blocks to module_list
        last_layer = 32
        for layer in layers:
            module_list.append(Block(last_layer, layer, stride=2))
            last_layer = layer
        self.cnn_net = torch.nn.Sequential(*module_list)

        # define GAP
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        # add a final linear layer for classification
        self.classifier = torch.nn.Linear(last_layer, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits.
        It is the output of the neural network.

        Args:
            inputs: batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            batch of logits. Dimensions: [batch, output_channels].
        """

        # compute the features
        outputs = self.cnn_net(inputs)

        # GAP
        outputs = self.gap(outputs)

        # flatten output and compute linear layer output
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)

        return outputs
