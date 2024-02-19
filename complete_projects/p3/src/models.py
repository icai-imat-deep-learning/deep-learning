# deep learning libraries
import torch

# own modules
from src.utils import get_dropout_random_indexes


class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        super().__init__()
        self.p: float = p
        self.inplace: bool = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """

        # get activated neurons
        dropout_indexes: torch.Tensor = get_dropout_random_indexes(inputs.shape, self.p)

        # define outputs depending on inplace
        if self.inplace:
            outputs = inputs
        else:
            outputs = inputs.clone()

        if self.training:
            # filter neurons
            outputs[dropout_indexes == 1] = 0

            # scale by factor during training
            if self.training:
                outputs /= 1 - self.p

        return outputs


class BatchNorm(torch.nn.Module):
    # TODO: implement
    pass


class CNNModel(torch.nn.Module):
    # TODO
    pass
