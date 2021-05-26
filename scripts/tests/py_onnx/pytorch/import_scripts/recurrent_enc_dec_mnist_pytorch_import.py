from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import caffe2.python.onnx.backend as backend
import onnx
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MNIST recurrent enc dec ONNX import example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--onnx-file', type=str, default="onnx_models/lstm_enc_dec_mnist.onnx",
                    help='Path of the onnx file to load')
parser.add_argument('-m', '--target-metric', type=str, default="",
                    help='Path to a file with a single value with the target metric to achieve')
args = parser.parse_args()

device = torch.device("cpu")

kwargs = {'batch_size': args.batch_size}

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, drop_last=False, **kwargs)

print(f"Going to load the ONNX model from \"{args.onnx_file}\"")
model = onnx.load(args.onnx_file)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print("Going to build the caffe2 model from ONNX model")
rep = backend.prepare(model, device="CPU") # or CPU
print("Caffe2 model built!")

# Inference with Caffe2 backend (only way to "import with pytorch")
loss = 0
total_samples = 0
to_torch = lambda x: torch.from_numpy(x)
for data, label in tqdm(data_loader):
    data, label = data.numpy(), label.numpy()  # Caffe2 backend works with numpy not torch.Tensor
    data = np.reshape(data, (data.shape[0], 28, 28))
    outputs = rep.run(data)._0
    loss += ((data - outputs)**2).sum() / (28*28)
    total_samples += data.shape[0]

final_mse = loss / total_samples
print(f"Results: mse loss = {final_mse:.5f}")

if args.target_metric != "":
    with open(args.target_metric, 'r') as mfile:
        target_metric_val = float(mfile.read())

    metrics_diff = abs(final_mse - target_metric_val)
    if metrics_diff > 0.001:
        print(f"Test failed: Metric difference too high target={target_metric_val}, pred={final_mse:.5f}")
        sys.exit(1)
    else:
        print(f"Test passed!: target={target_metric_val}, pred={final_mse:.5f}")
