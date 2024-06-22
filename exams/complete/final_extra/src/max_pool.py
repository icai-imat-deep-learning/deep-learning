# deep learning libraries
import torch
import torch.nn.functional as F

# other libraries
from typing import Optional, Any


def unfold_max_pool(
    inputs: torch.Tensor, kernel_size: int, stride: int, padding: int
) -> None:
    """
    This function computes the unfold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. For that, we
    recommend to include the channels into another dimension that will
    not be affected by the consequently operations.

    Args:
        inputs: inputs tensor. Dimensions: []
        kernel_size: _description_
        stride: _description_
        padding: _description_

    Returns:
        _description_
    """

    # shrink channel dim
    inputs_unfolded = inputs.flatten(0, 1).unsqueeze(1)

    # unfold
    inputs_unfolded: torch.Tensor = F.unfold(
        inputs_unfolded, kernel_size, stride=stride, padding=padding
    )

    return inputs_unfolded


def fold_max_pool(
    inputs: torch.Tensor, output_size: int, batch_size: int, stride: int, padding: int
) -> None:
    """
    This function computes the fold needed for the MaxPool2d.
    Since the maxpool only comute sthe max over single channel
    and not over all the channels, we need that the second dimension of
    our unfold tensors are data from only channel. To do that, we
    recommend that the fold version recovers the channel dimensions
    before executing the fold operation.

    Args:
        inputs: _description_
        output_size: _description_
        batch_size: _description_
        stride: _description_
        padding: _description_

    Returns:
        _description_
    """

    # unshrink channel dim
    inputs_folded = inputs.view(batch_size, -1, inputs.shape[1])

    # fold
    inputs_folded: torch.Tensor = F.fold(
        inputs_folded, output_size, 1, stride=stride, padding=padding
    )

    return inputs_folded


class MaxPool2dFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the MaxPool2d.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # unfold inputs and compute outputs
        inputs_unfolded: torch.Tensor = unfold_max_pool(
            inputs, kernel_size, stride, padding
        )

        # compute max positions
        outputs_unfolded: torch.Tensor
        outputs_unfolded, max_indexes = torch.max(inputs_unfolded, dim=1)

        # compute fold version
        output_size: int = inputs.shape[2] - kernel_size + 1
        outputs: torch.Tensor = fold_max_pool(
            outputs_unfolded, output_size, inputs.shape[0], stride, padding
        )

        # save elements for the backward
        ctx.save_for_backward(
            inputs, max_indexes, torch.tensor(stride), torch.tensor(padding)
        )

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """
        This method is the backward of the Maxout.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions:
                [batch, output dim].

        Returns:
            inputs gradients. Dimensions: [batch, input dim].
            gradients for the first weights. Dimensions:
                [output dim, input dim].
            gradients for the first bias. Dimensions: [output dim].
            gradient for the second weights. Dimensions: [output dim,
                input dim].
            gradient for the second bias. Dimensions: [output dim].
        """

        # load tensors from the forward
        (
            inputs,
            weights_first,
            bias_first,
            weights_second,
            bias_second,
            indexes,
        ) = ctx.saved_tensors

        # compute grad outputs for each branch
        grad_outputs_first: torch.Tensor = grad_output.clone()
        grad_outputs_second: torch.Tensor = grad_output.clone()
        grad_outputs_first[~indexes] = 0
        grad_outputs_second[indexes] = 0

        # compuye grad inputs
        grad_inputs_first = torch.matmul(grad_outputs_first, weights_first)
        grad_inputs_second = torch.matmul(grad_outputs_second, weights_second)
        grad_inputs = grad_inputs_first + grad_inputs_second

        # compute weights and bias gradients
        grad_weight_first = torch.matmul(inputs.T, grad_outputs_first).T
        grad_weight_second = torch.matmul(inputs.T, grad_outputs_second).T
        grad_bias_first = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_outputs_first
        ).squeeze(0)
        grad_bias_second = torch.matmul(
            torch.ones_like(inputs[:, 0]).unsqueeze(0), grad_outputs_second
        ).squeeze(0)

        return (grad_inputs, None, None, None)


class MaxPool2d(torch.nn.Module):
    """
    This is the class that represents the Maxout Layer.
    """

    kernel_size: int
    stride: int

    def __init__(
        self, kernel_size: int, stride: Optional[int], padding: int = 0
    ) -> None:
        """
        This method is the constructor of the Maxout layer.
        """

        # call super class constructor
        super().__init__()

        # set attributes value
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

        # save function
        self.fn = MaxPool2dFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.kernel_size, self.stride, self.padding)
