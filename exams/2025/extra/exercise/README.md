# Final 2025

---

Welcome to the **Final Exam of 2025**!  

## ✅ Allowed Resources

- **PyTorch Documentation**  
  https://pytorch.org/docs/stable/
- **PyTorch Forums**  
  https://discuss.pytorch.org/

**Note:** You may use a web browser (e.g., Google) **only** to navigate to the two links above. All other websites, generative AI tools (including GitHub Copilot), or past projects code are **strictly forbidden**.

You cannot use any class or function from torch.nn module, and operations must be done with indexing (e.g. `torch.where` is not allowed). When in doubt if a function is allowed, ask your teacher!

## Custom MaxPool2d (7 points)

Here you will have to implement a custom MaxPool2d without using loops. First, let's explain the custom MaxPool2d operation. Usually, MaxPool2d is computed without any depth, that means that the output always has the same number of channels that the input. However, here we want to merge the concept of groups with the MaxPool2d. Therefore, now the max operation will be computed from all the channels in the same group.

```
Normal MaxPool2d:
Inputs: [batch size, channels, height, width]
Outputs: [batch size, channels, height - kernel size + 1, width - kernel size + 1].
```

```
Custom MaxPool2d with 1 group:
Inputs: [batch size, channels, height, width]
Outputs: [batch size, 1, height - kernel size + 1, width - kernel size + 1].
```

```
Custom MaxPool2d with n group:
Inputs: [batch size, channels, height, width]
Outputs: [batch size, n, height - kernel size + 1, width - kernel size + 1].
```

You will have to implement this without using any loops or any function from the nn package (torch.where is not allowed either), besides unfold, fold and one_hot. You may want look at the following functions:

    fold, unfold, permute, one_hot, max

### `forward` (2.5 points)

Here you will have to code the forward method.

### `backward` (4.5 points)

Here you will have to code the backward method.