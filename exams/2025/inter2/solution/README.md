# Inter Option 2

This is the inter exam of 2025. For this exam you are only allowed to use pytorch documentation and pytorch forums. In order to access those you can use a browser as google but only access pytorch documentation and pytorch forums pages. Other web pages, generative AI tools or any other material as previous projects are forbidden.

Moreover, for this exam you cannot use any class or function from torch.nn module, and operations must be done with indexing (e.g. torch.where is not allowed).

# Conv2d (5 points)

In this exercise you will have to implement the Conv2d. In this case you cannot use the fold and unfold operations, neither the nn package. Only two for-loops are allowed, to iterate over the spatial dimensions. The rest of the operations should be solved with indexing and tensor operations (e.g. torch.sum).

For this you will have to fill the forward of the class written in src.conv. Link to pytorch documentation of the layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html.

# HuberLoss (5 points)

In this exercise you will have to implement the forward and backward of the HuberLoss. In this case you cannot use loops, the nn package or torch.where. An image of the loss is plotted in the following image. MSELoss in blue and HuberLoss in green.

![alt text](artifacts/image.png)

For this you will have to fill the forward of the class written in src.huber_loss. Link to pytorch documentation of the layer: https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html. In this case, we strongly recommend looking at the documentation since it will help significantly to replicate the functionality, since you will find there the formula this loss follows.

The points for this layer are divided in the following:

## Forward (2 points)

This function is contained in the src.huber_loss. You will have to implement the forward method without using the nn package or torch.where.

## Backward (3 points)

This function is contained in the src.huber_loss. You will have to implement the forward method without using the nn package or torch.where.