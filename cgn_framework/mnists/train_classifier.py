import os
import argparse
import repackage
repackage.up()

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from mnists.models.classifier import CNN
from mnists.dataloader import get_tensor_dataloaders, TENSOR_DATASETS

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # stats
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
    train_acc = 100. * correct / len(train_loader.dataset)
    return train_acc
  

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def main(args):
    # model and dataloader
    model = CNN()
    dl_train, dl_test = get_tensor_dataloaders(args.dataset, args.batch_size, combined=args.combined)

    # Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_accs = dict()
    test_accs = dict()
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, dl_train, optimizer, epoch)
        test_acc = test(model, device, dl_test)
        train_accs[epoch] = train_acc
        test_accs[epoch] = test_acc
        scheduler.step()
    
    # save the final model
    dataset_suffix = (args.dataset) if not args.combined else (args.dataset + "_combined")
    dataset_suffix += "_seed_" + str(args.seed) if args.seed is not None else ""
    save_path = f'mnists/experiments/classifier_{dataset_suffix}/weights/ckp_epoch_{args.epochs}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print('Saving model to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)

    # save the test accuracy
    save_path = f'mnists/experiments/classifier_{dataset_suffix}/test_accs.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print('Saving test accuracy to {}'.format(save_path))
    torch.save(test_accs, save_path)
    torch.save(train_accs, save_path.replace("test_accs", "train_accs"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=TENSOR_DATASETS,
                        help='Provide dataset name.')
    parser.add_argument('--combined', action='store_true',
                        help='If set, train on both original and generated data.')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')

    args = parser.parse_args()

    print(args)
    main(args)
