from __future__ import print_function
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Encoder
        self.gru_enc = nn.GRU(28, 128)
        # Decoder
        self.gru_dec = nn.GRU(28, 128)
        self.dense = nn.Linear(128, 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x2):
        # Encoder
        x = x.permute(1, 0, 2)
        enc_out, hidden_enc = self.gru_enc(x)
        # Decoder
        x2 = x2.permute(1, 0, 2)
        dec_out, hidden_dec = self.gru_dec(x2, hidden_enc)
        x = self.dense(dec_out)
        out = self.sigmoid(x)
        out = out.permute(1, 0, 2)
        return out


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_acc = 0
    current_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        data_dec = torch.nn.functional.pad(data, (0, 0, 1, 0))[:,:-1,:]  # Shifted data
        data_el_size = data_dec.size(1) * data.size(2)  # 28 * 28 for mnist
        optimizer.zero_grad()
        output = model(data, data_dec)
        loss = F.mse_loss(output, data, reduction='sum')  
        loss.backward()
        loss_acc += loss.item() / data_el_size
        current_samples += data.size(0)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_acc / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    current_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_dec = torch.nn.functional.pad(data, (0, 0, 1, 0))[:,:-1,:]  # Shifted data
            data_el_size = data.size(1) * data.size(2)  # 28 * 28 for mnist
            output = model(data, data_dec)
            test_loss += F.mse_loss(output, data, reduction='sum').item() / data_el_size
            current_samples += data.size(0)

    test_loss = test_loss / current_samples
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GRU encoder-decoder MNIST Example')
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
    parser.add_argument('--output-path', type=str, default="onnx_models/gru_enc_dec_mnist.onnx",
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

    # Create data preprocessing functions
    class remove_ch(object):
        ''' Custom transform to preprocess data'''
        def __call__(self, img):
            return img.view((28, 28))

    transform=transforms.Compose([
        transforms.ToTensor(),
        remove_ch()
        ])

    # Prepare data generators
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, drop_last=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)

    # In case of providing output metric file, store the test accuracy value
    if args.output_metric != "":
        with open(args.output_metric, 'w') as ofile:
            ofile.write(str(test_loss))

    # Save to ONNX file
    # Input Encoder
    dummy_input = torch.randn(args.batch_size, 28, 28, device=device)
    # Input decoder
    dummy_input2 = torch.randn(args.batch_size, 28, 28, device=device)
    torch.onnx._export(model, (dummy_input, dummy_input2), args.output_path, keep_initializers_as_inputs=True)

if __name__ == '__main__':
    main()
