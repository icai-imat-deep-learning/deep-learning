# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class MaxoutFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Maxout.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # compute forward
        outputs_first: torch.Tensor = torch.matmul(
            inputs, weights_first.T
        ) + bias_first.unsqueeze(0)
        outputs_second: torch.Tensor = torch.matmul(
            inputs, weights_second.T
        ) + bias_second.unsqueeze(0)

        # compute final outputs
        outputs: torch.Tensor = outputs_second.clone()
        indexes: torch.Tensor = outputs_second <= outputs_first
        outputs[indexes] = outputs_first[indexes]

        # save tensors for the backward
        ctx.save_for_backward(
            inputs, weights_first, bias_first, weights_second, bias_second, indexes
        )

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward of the maxout.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
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

        # compute input gradients
        grad_inputs = torch.matmul(grad_output, weights_first)
        grad_inputs[~indexes] = torch.matmul(grad_output, weights_second)[~indexes]

        # compute weights and bias gradients
        inputs_first: torch.Tensor = inputs.clone()
        inputs_second: torch.Tensor = inputs.clone()
        inputs_first[indexes] = 0
        inputs_second[~indexes] = 0
        grad_weight_first = torch.matmul(inputs_first.T, grad_output).T
        grad_weight_second = torch.matmul(inputs_second.T, grad_output).T
        grad_bias_first = torch.matmul(
            torch.ones_like(inputs_first[:, 0]).unsqueeze(0), grad_output
        ).squeeze(0)
        grad_bias_second = torch.matmul(
            torch.ones_like(inputs_first[:, 0]).unsqueeze(0), grad_output
        ).squeeze(0)

        # compute gradients
        grad_input: torch.Tensor = torch.ones_like(inputs)
        grad_input[inputs <= 0] = 0
        grad_input *= grad_output

        return (
            grad_input,
            grad_weight_first,
            grad_bias_first,
            grad_weight_second,
            grad_bias_second,
        )


class Maxout(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weights_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_first: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )
        self.weights_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias_second: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim)
        )

        # init parameters corectly
        self.reset_parameters()

        self.fn = MaxoutFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(
            inputs,
            self.weights_first,
            self.bias_first,
            self.weights_second,
            self.bias_second,
        )

    @torch.no_grad()
    def set_parameters(
        self,
        weights_first: torch.Tensor,
        bias_first: torch.Tensor,
        weights_second: torch.Tensor,
        bias_second: torch.Tensor,
    ) -> None:
        """
        _summary_

        Args:
            weights_first: _description_
            bias_first: _description_
            weights_second: _description_
            bias_second: _description_

        Returns:
            _description_
        """

        # set attributes
        self.weights_first = torch.nn.Parameter(weights_first)
        self.bias_first = torch.nn.Parameter(bias_first)
        self.weights_second = torch.nn.Parameter(weights_second)
        self.bias_second = torch.nn.Parameter(bias_second)

        return None

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weights_first, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights_second, a=math.sqrt(5))
        if self.bias_first is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights_first)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias_first, -bound, bound)
            torch.nn.init.uniform_(self.bias_second, -bound, bound)

        return None
