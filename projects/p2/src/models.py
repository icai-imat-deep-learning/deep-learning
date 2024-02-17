# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
import math
from typing import Any


class ReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO


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

        self.fn = ReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs)


class LinearFunction(torch.autograd.Function):
    """
    This class implements the forward and backward of the Linear layer.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is the forward pass of the Linear layer.

        Args:
            ctx: contex for saving elements for the backward.
            inputs: inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: bias tensor. Dimensions: [output dimension].

        Returns:
            outputs tensor. Dimensions: [batch, output dimension].
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            tuple of gradients, gradients of the inputs, the weights
                and the bias, in respective order.
                Inputs gradients dimension: [batch, input dimension].
                Weights gradients dimension:
                [output dimension, input dimension].
                Bias gradients dimension: [output dimension].
        """

        # TODO


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_dim))

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = LinearFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimenions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.bias)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # TODO

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: contex for loading elements needed in the backward.
            grad_output: outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            tuple of gradients of inputs, weights, bias and two None
                values (you should not return a gradient for padding
                and stride).
                Inputs gradients dimensions: [batch, input channels,
                height, width].
                Weights gradients dimensions: [output channels,
                input channels, kernel size, kernel size].
                Bias gradients dimensions: [output channels].
        """

        # TODO


class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: input dimension.
            output_channels: output dimension.
            kernel_size: kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


class Block(torch.nn.Module):
    """
    Neural net block composed of 3x(conv(kernel=3, padding=1) + ReLU).

    Attributes:
        net: sequential containing all the layers.
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

        # TODO

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

        # TODO


class CNNModel(torch.nn.Module):
    """
    Model constructed used Block modules.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        input_channels: int = 3,
        output_channels: int = 10,
    ) -> None:
        """
        Constructor of the class CNNModel.

        Args:
            layers: output channel dimensions of the Blocks.
            input_channels: input channels of the model.
        """

        # TODO

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

        # TODO
