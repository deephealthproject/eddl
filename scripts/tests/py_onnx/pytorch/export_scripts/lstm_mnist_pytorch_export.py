from __future__ import print_function
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, input_dim=28, output_dim=10):
        super(Net, self).__init__()
        # Build the model
        self.in_dense = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU())
        self.recurrent = nn.LSTM(32, 128, batch_first=True)
        self.out_dense = nn.Sequential(
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=1))

    def forward(self, x):
        embedded = self.in_dense(x)
        # embedded = [batch_size, 28, 32]

        lstm_out, (h, c) = self.recurrent(embedded)
        h = torch.squeeze(h, 0)
        # h = [batch_size, 128]

        dense_out = self.out_dense(h)
        # dense_out = [batch_size, output_dim]

        return dense_out  # Get last pred from seq


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    current_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view((-1, 28, 28))
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        current_samples += data.size(0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view((-1, 28, 28))
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_acc))

    return test_loss, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LSTM MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str, default="onnx_models/lstm_mnist.onnx",
                        help='Output path to store the onnx file')
    parser.add_argument('--output-metric', type=str, default="",
                        help='Output file path to store the metric value obtained in test set')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True})

    # Prepare data generators
    transform=transforms.Compose([transforms.ToTensor()])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        _, test_acc = test(model, device, test_loader)

    # In case of providing output metric file, store the test accuracy value
    if args.output_metric != "":
        with open(args.output_metric, 'w') as ofile:
            ofile.write(str(test_acc))

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 28, 28, device=device)
    torch.onnx._export(model, dummy_input, args.output_path, keep_initializers_as_inputs=True)

if __name__ == '__main__':
    main()
