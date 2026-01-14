# standard libraries
import math
from typing import Any

# 3pps
import torch
import torch.nn.functional as F


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
            ctx: Context for saving elements for the backward.
            inputs: Input tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*], same as inputs.
        """

        # save tensors for the backward
        ctx.save_for_backward(inputs)

        # compute forward
        outputs = inputs.clone()
        outputs[outputs <= 0] = 0

        return outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients. Dimensions: [*].

        Returns:
            Inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # load tensors from the forward
        (inputs,) = ctx.saved_tensors

        # compute gradients
        grad_input: torch.Tensor = torch.ones_like(inputs)
        grad_input[inputs <= 0] = 0
        grad_input *= grad_output

        return grad_input


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
            inputs: Inputs tensor. Dimensions: [*].

        Returns:
            Outputs tensor. Dimensions: [*] (same as the input).
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
            ctx: Contex for saving elements for the backward.
            inputs: Inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: Bias tensor. Dimensions: [output dimension].

        Returns:
            Outputs tensor. Dimensions: [batch, output dimension].
        """

        # save elements for backward
        ctx.save_for_backward(inputs, weight, bias)

        return torch.matmul(inputs, weight.T) + bias.unsqueeze(0)

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.

        Args:
            ctx: Context for loading elements from the forward.
            grad_output: Outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            Inputs gradients. Dimensions: [batch, input dimension].
            Weights gradients. Dimensions: [output dimension,
                input dimension].
            Bias gradients. Dimension: [output dimension].
        """

        # load elements from forward
        inputs, weight, bias = ctx.saved_tensors

        # compute gradients
        grad_inputs = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(inputs.T, grad_output).T
        grad_bias = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_output
        ).squeeze(0)

        return grad_inputs, grad_weight, grad_bias


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight torch parameter. Dimensions: [output dimension,
            input dimension].
        bias: Bias torch parameter. Dimensions: [output dimension].
        fn: Autograd function.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
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
            inputs: Inputs tensor. Dimensions: [batch, input dim].

        Returns:
            Outputs tensor. Dimensions: [batch, output dim].
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
            ctx: Context for saving elements for the backward.
            inputs: Inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: Weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: Bias of the layer. Dimensions: [output channels].
            padding: padding parameter.
            stride: stride parameter.

        Returns:
            Output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # save elements for the backward
        ctx.save_for_backward(
            inputs, weight, bias, torch.tensor(padding), torch.tensor(stride)
        )

        # unfold inputs and compute outputs
        inputs_unfolded: torch.Tensor = F.unfold(
            inputs, weight.shape[2], padding=padding, stride=stride
        )
        outputs_unfolded: torch.Tensor = torch.matmul(
            inputs_unfolded.transpose(1, 2),
            weight.view(weight.size(0), -1).t().unsqueeze(0),
        ).transpose(1, 2)

        # compute fold outputs
        output_width_height: int = inputs.shape[2] - weight.shape[2] + 1
        outputs: torch.Tensor = F.fold(
            outputs_unfolded, output_width_height, 1, padding=padding, stride=stride
        )

        return outputs + bias.view(1, bias.shape[0], 1, 1)

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        This is the backward of the layer.

        Args:
            ctx: Context for loading elements needed in the backward.
            grad_output: Outputs gradients. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]

        Returns:
            Inputs gradients. Dimensions: [batch, input channels,
                height, width].
            Weight gradients. Dimensions: [output channels,
                input channels, kernel size, kernel size].
            Bias gradients. Dimensions: [output channels].
            None.
            None.
        """

        # load saved tensors
        inputs, weight, bias, padding, stride = ctx.saved_tensors
        padding = padding.item()
        stride = stride.item()

        # compute unfolded versions
        inputs_unfolded: torch.Tensor = F.unfold(
            inputs, weight.shape[2], padding=padding, stride=stride
        )
        grad_output_unfolded: torch.Tensor = F.unfold(
            grad_output, 1, padding=padding, stride=stride
        )
        weights_unfolded: torch.Tensor = (
            weight.view(weight.size(0), -1).t().unsqueeze(0)
        )

        # compute inputs grad
        grad_inputs_unfolded: torch.Tensor = torch.matmul(
            weights_unfolded, grad_output_unfolded
        )
        grad_inputs: torch.Tensor = F.fold(
            grad_inputs_unfolded,
            inputs.shape[2],
            weight.shape[2],
            padding=padding,
            stride=stride,
        )

        # compute weights grad
        grad_weight: torch.Tensor = torch.matmul(
            inputs_unfolded, grad_output_unfolded.transpose(1, 2)
        )
        grad_weight = grad_weight.view(
            grad_weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
            grad_weight.shape[-1],
        )
        grad_weight = grad_weight.permute(0, 4, 1, 2, 3).sum(0)

        # compute bias grad
        grad_bias = torch.sum(torch.ones_like(grad_output) * grad_output, dim=(0, 2, 3))

        return grad_inputs, grad_weight, grad_bias, None, None


class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.

    Attributes:
        weight: Weight pytorch parameter. Dimensions: [output channels,
            input channels, kernel size, kernel size].
        bias: Bias torch parameter. Dimensions: [output channels].
        padding: Padding parameter.
        stride: Stride parameter.
        fn: Autograd function.
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
            input_channels: Input dimension.
            output_channels: Output dimension.
            kernel_size: Kernel size to use in the convolution.
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
            inputs: Inputs tensor. Dimensions: [batch, input channels,
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
        net: Sequential containing all the layers.
    """

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        """
        Constructor of the Block class. It is composed of
        3x(conv(kernel=3) + ReLU). Only the second conv
        will have stride. Use a Sequential for encapsulating all the
        layers. Clue: convs may have padding to fit into the correct
        dimensions.

        Args:
            input_channels: Input channels for Block.
            output_channels: Output channels for Block.
            stride: Stride only for the second convolution of the
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
        This method if the forward pass.

        Args:
            inputs: Inputs batch of tensors.
                Dimensions: [batch, input_channels, height, width].

        Returns:
            Outputs batch of tensors. Dimensions: [batch, output_channels,
                (height - 1)/stride + 1, (width - 1)/stride + 1].
        """

        return self.net(inputs)


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
            layers: Output channel dimensions of the Blocks.
            input_channels: Input channels of the model.
        """

        # call torch.nn.Module constructor
        super().__init__()

        # initialize module_list with a conv of kernel 7 a ReLU and a
        # max pooling of kernel 3
        module_list = [
            torch.nn.Conv2d(input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        # add 3 Blocks to module_list
        last_layer = 32
        for layer in hidden_sizes:
            module_list.append(Block(last_layer, layer, stride=2))
            last_layer = layer
        self.cnn_net = torch.nn.Sequential(*module_list)

        # define GAP
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        # add a final linear layer for classification
        self.classifier = torch.nn.Linear(last_layer, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of logits. It is the output of the
        neural network.

        Args:
            inputs: Inputs batch of images.
                Dimensions: [batch, channels, height, width].

        Returns:
            Outputs batch of logits. Dimensions: [batch,
                output_channels].
        """

        # compute the features
        outputs = self.cnn_net(inputs)

        # GAP
        outputs = self.gap(outputs)

        # flatten output and compute linear layer output
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)

        return outputs
