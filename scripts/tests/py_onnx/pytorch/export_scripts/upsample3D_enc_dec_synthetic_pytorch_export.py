from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_enc = self.encoder(x)
        out = self.decoder(x_enc)
        return out


# Prepare data loader
class Dummy_datagen:
    def __init__(self, batch_size=2, n_samples=6):
        # Shape: (n_samples=n_samples, ch=3, depth=16, height=16, width=16)
        self.samples = np.linspace(0, 1, n_samples*3*16*16*16).reshape((n_samples, 3, 16, 16, 16)).astype(np.float32)
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
            return self.samples[target:target+self.bs]
        raise StopIteration

    def reset(self):
        '''Reset the iterator'''
        self.curr_idx = 0


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_acc = 0
    current_samples = 0
    for batch_idx, data in enumerate(train_loader):
        data = torch.from_numpy(data)
        data = data.to(device)
        b, c, d, h, w = data.size()
        data_el_size = c * d * h * w
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data, reduction='sum')
        loss.backward()
        loss_acc += loss.item() / data_el_size
        current_samples += data.size(0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.samples),
                100. * batch_idx / len(train_loader), loss_acc / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    current_samples = 0
    with torch.no_grad():
        for data in test_loader:
            data = torch.from_numpy(data)
            data = data.to(device)
            output = model(data)
            b, c, d, h, w = data.size()
            data_el_size = c * d * h * w
            test_loss += F.mse_loss(output, data, reduction='sum').item() / data_el_size
            current_samples += data.size(0)

    test_loss = test_loss / current_samples
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch Conv3D+Upsample encoder-decoder with synthetic data example')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str, default="onnx_models/upsample3D_enc_dec_synthetic.onnx",
                        help='Output path to store the onnx file')
    parser.add_argument('--output-metric', type=str, default="",
                        help='Output file path to store the metric value obtained in test set')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create data generators
    train_loader = Dummy_datagen(args.batch_size)
    test_loader = Dummy_datagen(args.batch_size)

    # Train
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        train_loader.reset()
        test_loader.reset()

    # In case of providing output metric file, store the test accuracy value
    if args.output_metric != "":
        with open(args.output_metric, 'w') as ofile:
            ofile.write(str(test_loss))

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 3, 16, 16, 16, device=device)
    torch.onnx._export(model, dummy_input, args.output_path, keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()
