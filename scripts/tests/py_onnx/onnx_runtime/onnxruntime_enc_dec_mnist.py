import sys

import numpy as np
from tqdm import tqdm
import argparse
import torch
from torchvision import datasets, transforms
import onnx
import onnxruntime


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='MNIST encoder-decoder')
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    parser.add_argument('-m', '--target-metric', type=str, default="",
                        help='Path to a file with a single value with the target metric to achieve')
    parser.add_argument('--channel-last', action='store_true', default=False,
                        help='Change input shape from channel first to channel last')
    args = parser.parse_args()

    device = torch.device("cpu")

    kwargs = {'batch_size': args.batch_size}

    # Prepare data loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    # Print ONNX graph
    onnx_model = onnx.load(args.onnx_file)
    print(onnx.helper.printable_graph(onnx_model.graph))

    # Prepare ONNX runtime
    # Create a session with the onnx model
    session = onnxruntime.InferenceSession(args.onnx_file, None)
    enc_input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Inference with ONNX runtime
    total_mse = 0
    total_samples = 0
    for data, label in tqdm(data_loader):
        # Prepare data
        data = data.numpy()
        if args.channel_last:
            data = np.reshape(data, (data.shape[0], 28, 28, 1))
        else:
            data = np.reshape(data, (data.shape[0], 1, 28, 28))
        # Run model
        result = session.run([output_name], {enc_input_name: data})
        pred = np.squeeze(np.array(result), axis=0)
        total_mse += ((data - pred)**2).sum() / (data.size/data.shape[0])
        total_samples += len(data)

    final_mse = total_mse / total_samples
    print(f"Results: mse = {final_mse:.5f}")

    if args.target_metric != "":
        with open(args.target_metric, 'r') as mfile:
            target_metric_val = float(mfile.read())

        metrics_diff = abs(final_mse - target_metric_val)
        if metrics_diff > 0.001:
            print(f"Test failed: Metric difference too high target={target_metric_val}, pred={final_mse:.5f}")
            sys.exit(1)
        else:
            print(f"Test passed!: target={target_metric_val}, pred={final_mse:.5f}")

if __name__ == '__main__':
    main()
