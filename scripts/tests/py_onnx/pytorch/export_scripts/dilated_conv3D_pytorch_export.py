import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Prepare data loader

class Dummy_datagen:
    def __init__(self, batch_size=2, n_samples=6, num_classes=1):
        # Shape: (n_samples=n_samples, ch=3, depth=64, height=64, width=64)
        self.samples = np.linspace(
            0, 1, n_samples*3*64*64*64).reshape((n_samples, 3, 64, 64, 64)).astype(np.float32)
        # Shape: (n_samples=n_samples, dim=num_classes)
        self.labels = np.linspace(
            0, 1, n_samples*num_classes).reshape((n_samples, num_classes)).astype(np.float32)
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


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        n_features = -1
        return torch.reshape(x, (batch_size, n_features))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(3, 4, kernel_size=(5, 5, 5), padding=0, dilation=2),
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            nn.Conv3d(4, 4, kernel_size=(2, 2, 2), padding=0, dilation=3),
            nn.Conv3d(4, 4, kernel_size=(3, 3, 3), padding=0, dilation=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            Flatten(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    current_samples = 0
    loss_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction='sum')
        loss.backward()
        loss_acc += loss.item()
        current_samples += data.size(0)
        optimizer.step()
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.samples),
            100. * batch_idx / len(train_loader), loss_acc / current_samples))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    current_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = torch.from_numpy(data), torch.from_numpy(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += ((target - output)**2).sum()
            current_samples += data.size(0)

    test_loss = test_loss / current_samples
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return test_loss.item()  # Get loss value from pytorch loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch dilated Conv3D synthetic Example')

    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--output-path', type=str, default="onnx_models/dilated_conv3D_synthetic.onnx",
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
    test_loss = -1.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)

    # In case of providing output metric file, store the test accuracy value
    if args.output_metric != "":
        with open(args.output_metric, 'w') as ofile:
            ofile.write(str(test_loss))

    # Save to ONNX file
    dummy_input = torch.randn(args.batch_size, 3, 64, 64, 64, device=device)
    torch.onnx._export(model, dummy_input, args.output_path,
                       keep_initializers_as_inputs=True)


if __name__ == '__main__':
    main()
