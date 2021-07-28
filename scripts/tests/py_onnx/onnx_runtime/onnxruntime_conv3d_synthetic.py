import sys
import argparse

import numpy as np
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
import onnx
import onnxruntime


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Test 3d models with synthetic data')

    parser.add_argument('-b', '--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    parser.add_argument('-m', '--target-metric', type=str, default="",
                        help='Path to a file with a single value with the target metric to achieve')
    parser.add_argument('--data-size', type=int, default=16,
                        help='Size of the depth, height, and width dimensions of the synthetic data')
    args = parser.parse_args()

    # Load ONNX model
    onnx_model = onnx.load(args.onnx_file)

    # Prepare ONNX runtime
    # Create a session with the onnx model
    session = onnxruntime.InferenceSession(args.onnx_file, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Prepare data loader
    class Dummy_datagen:
        def __init__(self, batch_size=2, n_samples=6, num_classes=1, ch=3, d=16, h=16, w=16):
            # Shape: (n_samples=n_samples, ch=3, depth=16, height=16, width=16)
            self.samples = np.linspace(0, 1, n_samples*ch*d*h*w).reshape((n_samples, ch, d, h, w)).astype(np.float32)
            # Shape: (n_samples=n_samples, dim=num_classes)
            self.labels = np.linspace(0, 1, n_samples*num_classes).reshape((n_samples, num_classes)).astype(np.float32)
            self.curr_idx = 0  # Current index of the batch
            self.bs = batch_size

        def __iter__(self):
            return self

        def __len__(self):
            return int(self.samples.shape[0] / self.bs)

        def __next__(self):
            target = self.curr_idx
            self.curr_idx += self.bs
            if target <= self.samples.shape[0]-self.bs:
                return self.samples[target:target+self.bs], self.labels[target:target+self.bs]
            raise StopIteration

    total_mse = 0
    total_samples = 0
    for data, label in tqdm(Dummy_datagen(args.batch_size, d=args.data_size, h=args.data_size, w=args.data_size)):
        # Run model
        result = session.run([output_name], {input_name: data})
        pred = np.squeeze(np.array(result), axis=0)
        total_mse += ((label - pred)**2).sum()
        total_samples += len(data)

    final_mse = total_mse / total_samples
    print(f"Results: mse = {final_mse:.5f}")

    if args.target_metric != "":
        with open(args.target_metric, 'r') as mfile:
            target_metric_val = float(mfile.read())

        metrics_diff = abs(final_mse - target_metric_val)
        if metrics_diff > 0.01:
            print(f"Test failed: Metric difference too high target={target_metric_val}, pred={final_mse:.5f}")
            sys.exit(1)
        else:
            print(f"Test passed!: target={target_metric_val}, pred={final_mse:.5f}")


if __name__ == '__main__':
    main()
