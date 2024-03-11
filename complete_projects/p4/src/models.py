# deep learning libraries
import torch

# other libraries
import math
from typing import Any


class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # init hn
        hn: torch.Tensor = torch.empty(
            (inputs.shape[0], inputs.shape[1], weight_hh.shape[0])
        ).double()
        # indexes: torch.Tensor = torch.ones_like(hn)
        hn[:, 0, :] = (
            torch.matmul(inputs[:, 0, :], weight_ih.T)
            + torch.matmul(h0[0], weight_hh.T)
            + bias_ih
            + bias_hh
        )
        hn[:, 0, :][hn[:, 0, :] <= 0] = 0

        # get sequence length
        tau: int = inputs.shape[1]

        # iter over time stamps
        for t in range(1, tau):
            # compute next state
            hn[:, t, :] = (
                torch.matmul(inputs[:, t, :], weight_ih.T)
                + torch.matmul(hn[:, t - 1, :], weight_hh.T)
                + bias_ih
                + bias_hh
            )
            hn[:, t, :][hn[:, t, :] <= 0] = 0

        # save tensor for backward
        ctx.save_for_backward(inputs, h0, hn, weight_ih, weight_hh, bias_ih, bias_hh)

        return hn, hn[:, -1, :].unsqueeze(0)

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # load tensors from the forward
        inputs: torch.Tensor
        h0: torch.Tensor
        weight_ih: torch.Tensor
        weight_hh: torch.Tensor
        bias_ih: torch.Tensor
        bias_hh: torch.Tensor
        inputs, h0, hn, weight_ih, weight_hh, bias_ih, bias_hh = ctx.saved_tensors

        # init grads
        grad_hidden: torch.Tensor = torch.empty(
            (inputs.shape[0], inputs.shape[1], weight_hh.shape[0])
        ).double()
        grad_input: torch.Tensor = torch.empty_like(inputs)
        grad_h0: torch.Tensor = torch.empty_like(h0)
        grad_weight_ih: torch.Tensor = torch.zeros_like(weight_ih)
        grad_weight_hh: torch.Tensor = torch.zeros_like(weight_hh)
        grad_bias_ih: torch.Tensor = torch.zeros_like(bias_ih)
        grad_bias_hh: torch.Tensor = torch.zeros_like(bias_hh)

        # last time stamp
        grad_hidden[:, -1, :] = grad_output[:, -1, :]
        grad_hidden[:, -1, :][hn[:, -1, :] <= 0] = 0
        grad_input[:, -1, :] = torch.matmul(grad_hidden[:, -1, :], weight_ih)
        grad_weight_ih += torch.matmul(grad_hidden[:, -1, :].T, inputs[:, -1, :])
        grad_weight_hh += torch.matmul(grad_hidden[:, -1, :].T, hn[:, -2, :])
        grad_bias_ih += torch.matmul(
            torch.ones_like(inputs[:, 0, :]).T, grad_hidden[:, -1, :]
        ).sum(0)
        grad_bias_hh += torch.matmul(
            torch.ones_like(hn[:, 0, :]).T, grad_hidden[:, -1, :]
        ).sum(0)

        # get sequence length
        tau: int = inputs.shape[1]

        # iter over time stamps
        for t in range(2, tau):
            grad_hidden[:, -t, :] = (
                torch.matmul(grad_hidden[:, -(t - 1), :], weight_hh)
                + grad_output[:, -t, :]
            )
            grad_hidden[:, -t, :][hn[:, -t, :] <= 0] = 0
            grad_input[:, -t, :] = torch.matmul(grad_hidden[:, -t, :], weight_ih)
            grad_weight_ih += torch.matmul(grad_hidden[:, -t, :].T, inputs[:, -t, :])
            grad_weight_hh += torch.matmul(grad_hidden[:, -t, :].T, hn[:, -(t + 1), :])
            grad_bias_ih += torch.matmul(
                torch.ones_like(inputs[:, 0, :]).T, grad_hidden[:, -t, :]
            ).sum(0)
            grad_bias_hh += torch.matmul(
                torch.ones_like(hn[:, 0, :]).T, grad_hidden[:, -t, :]
            ).sum(0)

        # compute for first time stamp
        grad_hidden[:, -tau, :] = (
            torch.matmul(grad_hidden[:, -(tau - 1), :], weight_hh)
            + grad_output[:, -tau, :]
        )
        grad_hidden[:, -tau, :][hn[:, -tau, :] <= 0] = 0
        grad_input[:, -tau, :] = torch.matmul(grad_hidden[:, -tau, :], weight_ih)
        grad_weight_ih += torch.matmul(grad_hidden[:, -tau, :].T, inputs[:, -tau, :])
        grad_weight_hh += torch.matmul(grad_hidden[:, -tau, :].T, h0[0])
        grad_bias_ih += torch.matmul(
            torch.ones_like(inputs[:, 0, :]).T, grad_hidden[:, -tau, :]
        ).sum(0)
        grad_bias_hh += torch.matmul(
            torch.ones_like(hn[:, 0, :]).T, grad_hidden[:, -tau, :]
        ).sum(0)

        # compute h0
        grad_h0[0] = torch.matmul(grad_hidden[:, -(tau), :], weight_hh)

        return (
            grad_input,
            grad_h0,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )


class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """

        # call super class constructor
        super().__init__()

        self.hidden_size = hidden_size

        # define rnn
        self.rnn = torch.nn.LSTM(
            input_size=24, hidden_size=self.hidden_size, num_layers=2, batch_first=True
        )

        # define mlp
        self.mlp = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Linear(self.hidden_size, 24)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """

        # define inputs to rnn
        h0: torch.Tensor = torch.zeros(2, inputs.size(0), self.hidden_size).to(
            self.rnn.bias_hh_l0.device
        )
        c0: torch.Tensor = torch.zeros(2, inputs.size(0), self.hidden_size).to(
            self.rnn.bias_hh_l0.device
        )

        # compute outputs
        outputs, _ = self.rnn(inputs, (h0, c0))
        outputs = self.mlp(outputs)

        return outputs
