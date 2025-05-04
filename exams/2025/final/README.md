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

## MaxOut (5 points)

You have to code MaxOut layer in pytorch for a generic number of units. For this implementation you cannot use loops, so everything has to be vectorized operations. You have the formula and diagram of the MaxOut in the `artifacts` folder. Therefore, we will use the [batch matrix multiplication](https://pytorch.org/docs/stable/generated/torch.bmm.html) (BMM). We advice to have a look at it, since understanding the formula and shapes is a key step to complete the exercise. In this case you have to complete the following functions:

### `__init__` (0.5 points)

Here you will have to define the constructor and the objects that will be used in next methods.

### `reshape_inputs` (1 point)

This method reshapes the inputs in a way they can be used in the BMM. For this function the target shape is provided in the docstring of the function. A test is available to check the functionality of the method. 

Hint: Take into account that for the reshaping you may need to use repetitions (such as [`torch.repeat_interleave()`](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html)) and other functions, and not only `.view()` operations.

### `reshape_weight` (2 points)

This method reshapes the weight tensor created in the init method in a way that it can be used in the BMM. For this function the target shape is not provided and there is not an available test. You can figure out if the implementation is correct in the test for the `forward` pass. 

Hint: Take into account that for the reshaping you may have to use repetitions and other functions, and not only `.view()` operations.

### `forward` (1.5 points)

This method is the complete forward pass. You will have to use the two previous functions and the BMM to perform the MaxOut. 

--- 
Good luck, and happy coding!
