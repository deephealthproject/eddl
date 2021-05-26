import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Generate random image
# image = torch.randn((1, 3, 5, 5), dtype=torch.float)

# Fix input image
image = torch.tensor(
    [[[[-2.13, 0.60, 0.81, -1.50, -0.55],
       [-0.02, 0.61, 0.95, -1.18, 0.75],
       [0.55, -1.22, -0.62, -0.13, 0.21],
       [0.06, -1.16, -1.14, -0.60, 1.42],
       [3.27, -0.47, -0.86, 0.27, -0.26]],

      [[-0.15, 1.23, 1.21, 2.31, 2.23],
       [-0.04, -0.35, -0.53, -0.73, -0.17],
       [-1.03, 1.03, -1.24, 0.36, -0.52],
       [-0.81, 0.59, 1.90, 0.06, -0.85],
       [0.11, -0.09, 0.84, 0.14, -1.68]],

      [[-0.23, 0.33, 0.48, -1.43, -0.05],
       [-0.10, -0.13, -1.32, 0.07, -0.35],
       [0.61, 1.96, 1.11, 0.60, -0.83],
       [0.58, 0.28, 0.32, -0.05, -1.93],
       [0.03, -0.46, 0.08, 0.49, 0.39]]]]
)
layer = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))

# Fix params
layer.bias = nn.Parameter(torch.zeros((1,), dtype=torch.float))
layer.weight = nn.Parameter(torch.ones((3, 1, 3, 3), dtype=torch.float))

# Get output
output = layer(image)

# From device to cpu and numpy
a = image.cpu().detach().numpy()
b = output.cpu().detach().numpy()

# Print tensors
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("Input:")
print(a)
print("-------------------")
print("Output:")
print(b)
