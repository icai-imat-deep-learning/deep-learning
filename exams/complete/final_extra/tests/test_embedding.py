# deep learning libraries
import torch

# other libraries
import pytest

# own modules
from src.embedding import Embedding
from src.utils import set_seed


@pytest.mark.order(3)
def test_embedding() -> None:
    for seed in range(10):
        set_seed(seed)
        inputs: torch.Tensor = torch.randint(high=21, size=(64,))

        # define models
        set_seed(seed)
        model: torch.nn.Module = Embedding(64, 20, padding_idx=2)
        set_seed(seed)
        model_torch: torch.nn.Module = torch.nn.Embedding(64, 20, padding_idx=2)

        # compute backward of our maxpool
        outputs = model(inputs)
        # if inputs.grad is not None:
        #     inputs.grad.zero_()
        model.zero_grad()
        outputs.sum().backward()
        if model.weight.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_weight: torch.Tensor = model.weight.grad.clone()

        # compute backward of pytorch maxpool
        outputs_torch = model_torch(inputs)
        model_torch.zero_grad()
        outputs_torch.sum().backward()
        if model_torch.weight.grad is None:
            assert False, "Gradients not returned, none value detected"
        grad_weight_torch: torch.Tensor = model_torch.weight.grad.clone()

        # check grad weight size
        assert grad_weight.shape == grad_weight_torch.shape, (
            f"Incorrect grad inputs shape, expected {grad_weight_torch.shape}, got "
            f"{grad_weight.shape}"
        )

        # check grad weight
        assert torch.allclose(
            grad_weight, grad_weight_torch, atol=1e-10
        ), "Incorrect grad inputs"

    return None
