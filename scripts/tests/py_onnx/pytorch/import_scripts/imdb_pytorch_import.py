from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import caffe2.python.onnx.backend as backend
import onnx
from tqdm import tqdm
from torchtext.legacy import datasets, data

parser = argparse.ArgumentParser(description='PyTorch IMDB import ONNX example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--onnx-file', type=str, default="onnx_models/lstm_imdb.onnx",
                    help='Path of the onnx file to load')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--vocab-size', type=int, default=2000,
                    help='Max size of the vocabulary (default: 2000)')
parser.add_argument('-uin', '--unsqueeze-input', action='store_true', default=False,
                    help='Input shape from [batch, seq_len] to [batch, seq_len, 1]')
parser.add_argument('-m', '--target-metric', type=str, default="",
                    help='Path to a file with a single value with the target metric to achieve')
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device("cpu")

# Create data fields for preprocessing
TEXT = data.Field()
LABEL = data.LabelField()
# Create data splits
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
# Create vocabulary
TEXT.build_vocab(train_data, max_size = args.vocab_size-2)
LABEL.build_vocab(train_data)
# Create splits iterators
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = args.batch_size,
    device = device)

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
for batch in tqdm(test_iterator):
    data, label = batch.text.numpy(), batch.label.float()
    if args.unsqueeze_input:
        data = torch.unsqueeze(data, -1)
    data = np.transpose(data, (1, 0))
    label = label.view(label.shape + (1,))
    outputs = torch.tensor(rep.run(data))
    pred = torch.round(outputs)
    correct += pred.eq(label).sum().item()
    total += len(label)

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
