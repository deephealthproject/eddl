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

parser = argparse.ArgumentParser(description='PyTorch MNIST ONNX import example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--onnx-file', type=str, default="onnx_models/conv2D_mnist.onnx",
                    help='Path of the onnx file to load')
parser.add_argument('--input-1D', action='store_true', default=False,
                    help='To change the input size to a 784 length vector')
parser.add_argument('--no-channel', action='store_true', default=False,
                    help='If --input-1D is enabled, removes the channel dimension. (bs, 1, 784) -> (bs, 784)')
parser.add_argument('-m', '--target-metric', type=str, default="",
                    help='Path to a file with a single value with the target metric to achieve')
args = parser.parse_args()

device = torch.device("cpu")

kwargs = {'batch_size': args.batch_size}

class remove_channel_dim(object):
    ''' Custom transform to preprocess data'''
    def __call__(self, img):
        return torch.squeeze(img)

if args.no_channel:
    class from2Dto1D(object):
        ''' Custom transform to preprocess data'''
        def __call__(self, img):
            return img.view((-1))
else:
    class from2Dto1D(object):
        ''' Custom transform to preprocess data'''
        def __call__(self, img):
            return img.view((1, -1))

# Prepare data preprocessing
_transforms = [transforms.ToTensor()]
if args.input_1D:
    _transforms.append(from2Dto1D())
elif args.no_channel:
    _transforms.append(remove_channel_dim())
    
transform = transforms.Compose(_transforms)

# Create data generator
dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transform)
data_loader = torch.utils.data.DataLoader(dataset, drop_last=True, **kwargs)

print(f"Going to load the ONNX model from \"{args.onnx_file}\"")
model = onnx.load(args.onnx_file)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

print("Going to build the caffe2 model from ONNX model")
rep = backend.prepare(model, device="CPU") # or CPU
print("Caffe2 model built!")

# Inference with Caffe2 backend (only way to "import with pytorch")
correct = 0
total = 0
for data, label in tqdm(data_loader):
    data, label = data.numpy(), label.numpy()
    outputs = rep.run(data)
    prediction = np.array(np.argmax(np.array(outputs).squeeze(), axis=1).astype(np.int))
    correct += np.sum(prediction == label)
    total += len(prediction)

final_acc = correct / total
print(f"Results: Accuracy = {final_acc*100:.2f}({correct}/{total})")

if args.target_metric != "":
    with open(args.target_metric, 'r') as mfile:
        target_metric_val = float(mfile.read())

    metrics_diff = abs(final_acc - target_metric_val)
    if metrics_diff > 0.001:
        print(f"Test failed: Metric difference too high target={target_metric_val}, pred={final_acc:.5f}")
        sys.exit(1)
    else:
        print(f"Test passed!: target={target_metric_val}, pred={final_acc:.5f}")
