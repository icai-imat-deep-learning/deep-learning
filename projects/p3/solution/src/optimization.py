"""
This module contains the code to implement optimizers.
"""

# Standard libraries
from typing import Iterator, Dict, Any, DefaultDict

# 3pps
import torch


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # iter over groups
        for group in self.param_groups:
            # get parameters
            lr: torch.Tensor = group["lr"]
            weight_decay: torch.Tensor = group["weight_decay"]

            # iter over parameters
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # compute parameter grad
                parameter_grad: torch.Tensor = (
                    parameter.grad.data + weight_decay * parameter.data
                )

                # compute step parameter
                parameter.data -= lr * parameter_grad

        return None


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """

        # iter over groups
        for group in self.param_groups:
            # get parameters
            lr: torch.Tensor = group["lr"]
            momentum: torch.Tensor = group["momentum"]
            weight_decay: torch.Tensor = group["weight_decay"]

            # iter over parameters
            parameter: torch.Tensor
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # compute parameter grad
                parameter_grad: torch.Tensor = parameter.grad.data + (
                    weight_decay * parameter.data
                )

                # get parameter state
                param_state: Dict[str, torch.Tensor] = self.state[parameter]

                # init momentum
                if "momentum" not in param_state:
                    param_state["momentum"] = torch.zeros_like(parameter.data)

                # update momentum
                param_state["momentum"] = (
                    momentum * param_state["momentum"]
                ) + parameter_grad

                # compute step parameter
                parameter.data -= lr * param_state["momentum"]

        return None


class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # iter over groups
        for group in self.param_groups:
            # get parameters
            lr: torch.Tensor = group["lr"]
            momentum: torch.Tensor = group["momentum"]
            weight_decay: torch.Tensor = group["weight_decay"]

            # iter over parameters
            parameter: torch.Tensor
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # compute parameter grad
                parameter_grad: torch.Tensor = parameter.grad.data + (
                    weight_decay * parameter.data
                )

                # get parameter state
                param_state: Dict[str, torch.Tensor] = self.state[parameter]

                # init momentum
                if "momentum" not in param_state:
                    param_state["momentum"] = torch.zeros_like(parameter.data)

                # update momentum
                param_state["momentum"] = (
                    momentum * param_state["momentum"]
                ) + parameter_grad

                # compute step parameter
                parameter.data -= lr * (
                    parameter_grad + momentum * param_state["momentum"]
                )

        return None


class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # iter over groups
        for group in self.param_groups:
            # get parameters
            b1: torch.Tensor
            b2: torch.Tensor
            b1, b2 = group["betas"]
            lr: torch.Tensor = group["lr"]
            weight_decay: torch.Tensor = group["weight_decay"]
            eps = group["eps"]

            # iter over parameters
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                # compute parameter grad
                parameter_grad: torch.Tensor = parameter.grad.data + (
                    weight_decay * parameter.data
                )

                # get parameter state
                param_state: Dict[str, torch.Tensor] = self.state[parameter]

                # init parameters
                if "t" not in param_state:
                    param_state["mt"] = torch.zeros_like(parameter.data)
                    param_state["vt"] = torch.zeros_like(parameter.data)
                    param_state["t"] = torch.tensor(0.0)

                # compute mt and vt
                param_state["mt"] = b1 * param_state["mt"] + (1 - b1) * parameter_grad
                param_state["vt"] = b2 * param_state["vt"] + (1 - b2) * (
                    parameter_grad**2
                )

                # compute hat variables
                param_state["t"] += 1
                mt_hat: torch.Tensor = param_state["mt"] / (1 - b1 ** param_state["t"])
                vt_hat: torch.Tensor = param_state["vt"] / (1 - b2 ** param_state["t"])

                # compute step parameter
                parameter.data -= lr * (mt_hat / (torch.sqrt(vt_hat) + eps))

        return None
