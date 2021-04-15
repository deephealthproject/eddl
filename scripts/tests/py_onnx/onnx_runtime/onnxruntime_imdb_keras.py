import sys

import torch
import argparse
import os
import numpy as np
import onnx
import keras2onnx
import onnxruntime
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Training settings
parser = argparse.ArgumentParser(description='Inference with ONNX runtime from Keras ONNX model')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='Input batch size (default: 10)')
parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                    help='File path to the onnx file with the pretrained model to test')
parser.add_argument('--max-len', type=int, default=80,
                    help='Max len of input sequences (default: 80)')
parser.add_argument('--max-features', type=int, default=2000,
                    help='Maximum number of words from IMDB dataset')
parser.add_argument('-uin', '--unsqueeze-input', action='store_true', default=False,
                    help='Input shape from [batch, seq_len] to [batch, seq_len, 1]')
parser.add_argument('-m', '--target-metric', type=str, default="",
                    help='Path to a file with a single value with the target metric to achieve')
args = parser.parse_args()


print('Loading data...')
_, (x_test, y_test) = imdb.load_data(num_words=args.max_features)
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_test = sequence.pad_sequences(x_test, maxlen=args.max_len)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

def batch_gen(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size].astype(np.float32), y[i:i+batch_size]

# Prepare ONNX runtime
session = onnxruntime.InferenceSession(args.onnx_file, None)  # Create a session with the onnx model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
   
# Inference with ONNX runtime
correct = 0
total = 0
for data, labels in batch_gen(x_test, y_test, args.batch_size):
    if args.unsqueeze_input:
        data = data.reshape(data.shape + (1,))
    output = session.run([output_name], {input_name: data})
    pred = torch.round(torch.tensor(output)).reshape((args.batch_size))
    correct += pred.eq(torch.tensor(labels)).sum().item()
    total += args.batch_size

final_acc = correct / total
print(f"Results: Accuracy = {final_acc*100:.2f}({correct}/{total})")

if args.target_metric != "":
    with open(args.target_metric, 'r') as mfile:
        target_metric_val = float(mfile.read())

    metrics_diff = abs(final_acc - target_metric_val)
    if metrics_diff > 0.05:  # The dataset is not the same, we look for similar accuracy
        print(f"Test failed: Metric difference too high target={target_metric_val}, pred={final_acc:.5f}")
        sys.exit(1)
    else:
        print(f"Test passed!: target={target_metric_val}, pred={final_acc:.5f}")
