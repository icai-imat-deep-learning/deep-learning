# deep learning libraries
import torch

# other libraries
from typing import Any, Optional


class SoftshrinkFuncion(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Embeddings.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        lambd: float,
    ) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        # compute embeddings
        outputs: torch.Tensor = torch.zeros_like(inputs)
        outputs[inputs > lambd] = inputs[inputs > lambd] - lambd
        outputs[inputs < -lambd] = inputs[inputs < -lambd] + lambd

        # save tensors for the backward
        ctx.save_for_backward(inputs, torch.tensor(lambd))

        return outputs

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
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
        (inputs, lambd_tensor) = ctx.saved_tensors
        lambd: int = lambd_tensor.item()

        # compute gradients
        grad_inputs: torch.Tensor = torch.ones_like(inputs)
        grad_inputs[(inputs >= -lambd) & (inputs <= lambd)] = 0
        grad_inputs *= grad_outputs

        return grad_inputs, None


class Softshrink(torch.nn.Module):
    """
    This is the class that represents the Embedding Layer.
    """

    padding_idx: int

    def __init__(self, lambd: float = 0.5) -> None:
        """
        This method is the constructor of the Embedding layer.
        """

        # call super class constructor
        super().__init__()

        # init parameters corectly
        self.lambd = lambd

        self.fn = SoftshrinkFuncion.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.lambd)
