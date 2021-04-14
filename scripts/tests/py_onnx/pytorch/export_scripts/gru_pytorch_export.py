from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.legacy import datasets, data


class Net(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=32, output_dim=1):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # Build the model
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.recurrent = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            )

    def forward(self, x):
        embedded = self.embedding(x)  
        # embedded = [n_seq, batch_size, n_embedding]

        gru_out, h = self.recurrent(embedded)
        h = torch.squeeze(h, 0)
        # h = [batch_size, n_hidden]

        dense_out = self.dense(h)
        # dense_out = [batch_size, output_dim]

        return dense_out  # Get last pred from seq


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    correct = 0
    current_samples = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = batch.text, batch.label.float()
        data = torch.transpose(data, 0, 1)
        target = target.view((target.shape[0], 1))
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        pred = torch.round(output)
        correct += pred.eq(target).sum().item()
        current_samples += len(target)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                100. * correct / current_samples))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch.text, batch.label.float()
            data = torch.transpose(data, 0, 1)
            target = target.view((target.shape[0], 1))
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.round(output)
            correct += pred.eq(target).sum().item()

    test_acc = correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * test_acc))

    return test_loss, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch IMDB GRU Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--vocab-size', type=int, default=2000,
                        help='Max size of the vocabulary (default: 2000)')
    parser.add_argument('--output-path', type=str, default="onnx_models/gru_imdb.onnx",
                        help='Output path to store the onnx file')
    parser.add_argument('--output-metric', type=str, default="",
                        help='Output file path to store the metric value obtained in test set')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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

    model = Net(input_dim=len(TEXT.vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_iterator, criterion, optimizer, epoch)
        _, test_acc = test(model, device, test_iterator, criterion)

    # In case of providing output metric file, store the test accuracy value
    if args.output_metric != "":
        with open(args.output_metric, 'w') as ofile:
            ofile.write(str(test_acc))

    # Save to ONNX file
    dummy_input = torch.zeros((args.batch_size, 1000)).long().to(device)
    torch.onnx._export(model, dummy_input, args.output_path, keep_initializers_as_inputs=True, input_names=["input_1"])

if __name__ == '__main__':
    main()
