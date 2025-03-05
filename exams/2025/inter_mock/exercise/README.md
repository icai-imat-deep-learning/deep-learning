# Inter Model 1

This is the inter exam of 2025. For this exam you are only allowed to use pytorch documentation and pytorch forums. In order to access those you can use a browser as google but only access pytorch documentation and pytorch forums pages. Other web pages, generative AI tools or any other material as previous projects are forbidden.

Moreover, for this exam you cannot use loops, any class or function from torch.nn module, and operations must be done with indexing (e.g. torch.where is not allowed).

## GroupNorm (6 points)

In this exercise you have to replicate the torch GroupNorm layer. Take into account that this is a basic version of it, so you won't have to implement the affine transformation (scaling parameters).

For this you will have to fill the forward of the class written in src.group_norm. Link to pytorch documentation of the layer: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html.


## Hardshrink (4 points)

In this exercise you will have to implement the forward and backward of the Hardshrink layer. 

For this you will have to fill the methods of the class written in src.hardshrink. Link to pytorch documentation of the layer: https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html.

### Hardshrink Forward (1.5 points)

This function is contained in the src.hardshrink.

### Hardshrink Backward (2.5 points)

This function is contained in the src.hardshrink.