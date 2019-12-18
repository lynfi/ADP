'''Train CIFAN10 with PyTorch. G2'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from matplotlib import pyplot as plt
import copy

import vgg
from utils import progress_bar
from ADPF import *
from normal import *
from readmodel import *

parser = argparse.ArgumentParser(description='PyTorch CIFAN10 Training')
parser.add_argument('--name', default='RSE', type=str, help='RSE, SAP')
parser.add_argument('--gpu',
                    default=None,
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--epoch',
                    default=200,
                    type=int,
                    help='total epochs to run')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')

args = parser.parse_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # best test accuracy

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=100,
                                         shuffle=False,
                                         num_workers=2,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')
# Model
print('==> Building model..')

net1 = vgg.VGG('VGG16', nclass=10, keep_ratio=1.)
net2 = vgg.VGG('VGG16', nclass=10, keep_ratio=1.)
net3 = vgg.VGG('VGG16', nclass=10, keep_ratio=1.)
net1 = nn.Sequential(NormalizeLayer(), net1)
net2 = nn.Sequential(NormalizeLayer(), net2)
net3 = nn.Sequential(NormalizeLayer(), net3)

cudamodel(net1, device)
cudamodel(net2, device)
cudamodel(net3, device)

if device == 'cuda':
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    ckpname = ('./checkpoint/ADP_' + args.name + '.pth')
    loaddivmodel(net1, 'net1', ckpname)
    loaddivmodel(net2, 'net2', ckpname)
    loaddivmodel(net3, 'net3', ckpname)
    checkpoint = torch.load(ckpname)
    start_epoch = checkpoint['epoch'] + 1

criterion = nn.CrossEntropyLoss()
NLLloss = nn.NLLLoss()
optimizer_net1 = optim.SGD(net1.parameters(),
                           lr=args.lr,
                           weight_decay=2e-4,
                           momentum=.9)
optimizer_net2 = optim.SGD(net2.parameters(),
                           lr=args.lr,
                           weight_decay=2e-4,
                           momentum=.9)
optimizer_net3 = optim.SGD(net3.parameters(),
                           lr=args.lr,
                           weight_decay=2e-4,
                           momentum=.9)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)

    net1.train()
    net2.train()
    net3.train()

    train_loss_net1 = 0
    train_loss_net2 = 0
    train_loss_net3 = 0
    train_ADP = 0

    correct_net1 = 0
    correct_net2 = 0
    correct_net3 = 0
    alpha = 2
    beta = .5
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_net1.zero_grad()
        optimizer_net2.zero_grad()
        optimizer_net3.zero_grad()

        outputs_net1 = net1(inputs)
        outputs_net2 = net2(inputs)
        outputs_net3 = net3(inputs)

        loss_net1 = criterion(outputs_net1, targets)
        loss_net2 = criterion(outputs_net2, targets)
        loss_net3 = criterion(outputs_net3, targets)

        CEloss = (loss_net1 + loss_net2 + loss_net3)

        outputs_all = torch.stack([outputs_net1, outputs_net2, outputs_net3])
        ADP = alpha * Ensemble_Entropy(outputs_all).mean() + beta * log_det(
            targets, outputs_all).mean()
        loss = CEloss - ADP

        loss.backward()
        optimizer_net1.step()
        optimizer_net2.step()
        optimizer_net3.step()

        train_loss_net1 += loss_net1.item()
        train_loss_net2 += loss_net2.item()
        train_loss_net3 += loss_net3.item()
        train_ADP += ADP.item()

        total += targets.size(0)

        _, predicted = outputs_net1.max(1)
        correct_net1 += predicted.eq(targets).sum().item()

        _, predicted = outputs_net2.max(1)
        correct_net2 += predicted.eq(targets).sum().item()

        _, predicted = outputs_net3.max(1)
        correct_net3 += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx, len(trainloader),
            'ADP: %.2f | N1: %.2f (%.1f%%) | N2: %.2f (%.1f%%) | N3: %.2f (%.1f%%) '
            % (train_ADP / (batch_idx + 1), train_loss_net1 /
               (batch_idx + 1), 100. * correct_net1 / total, train_loss_net2 /
               (batch_idx + 1), 100. * correct_net2 / total, train_loss_net3 /
               (batch_idx + 1), 100. * correct_net3 / total))

    return (train_loss_net1 / (batch_idx + 1),
            train_loss_net2 / (batch_idx + 1),
            train_loss_net3 / (batch_idx + 1), 100. * correct_net1 / total,
            100. * correct_net2 / total, 100. * correct_net3 / total)


def test(epoch):
    net1.eval()
    net2.eval()
    net3.eval()

    test_loss_net1 = 0
    test_loss_net2 = 0
    test_loss_net3 = 0
    test_loss_ensemble = 0
    logdet = 0
    entropy = 0

    correct_net1 = 0
    correct_net2 = 0
    correct_net3 = 0
    correct_ensemble = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_net1 = net1(inputs)
            outputs_net2 = net2(inputs)
            outputs_net3 = net3(inputs)

            outputs_all = torch.stack(
                [outputs_net1, outputs_net2, outputs_net3])

            outputs_ensemble = F.softmax(outputs_all, dim=2).mean(0)

            logdet += log_det(targets, outputs_all).mean().item()
            entropy += Ensemble_Entropy(outputs_all).mean().item()

            loss_net1 = criterion(outputs_net1, targets)
            loss_net2 = criterion(outputs_net2, targets)
            loss_net3 = criterion(outputs_net3, targets)
            loss_ensemble = NLLloss(torch.log(outputs_ensemble + 1e-10),
                                    targets)

            test_loss_net1 += loss_net1.item()
            test_loss_net2 += loss_net2.item()
            test_loss_net3 += loss_net3.item()
            test_loss_ensemble += loss_ensemble.item()

            total += targets.size(0)

            _, predicted = outputs_net1.max(1)
            correct_net1 += predicted.eq(targets).sum().item()

            _, predicted = outputs_net2.max(1)
            correct_net2 += predicted.eq(targets).sum().item()

            _, predicted = outputs_net3.max(1)
            correct_net3 += predicted.eq(targets).sum().item()

            _, predicted = outputs_ensemble.max(1)
            correct_ensemble += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(testloader),
                'N1: %.2f (%.1f%%) | N2: %.2f (%.1f%%) | N3: %.2f (%.1f%%) | E: %.2f (%.1f%%)'
                %
                (test_loss_net1 /
                 (batch_idx + 1), 100. * correct_net1 / total, test_loss_net2 /
                 (batch_idx + 1), 100. * correct_net2 / total, test_loss_net3 /
                 (batch_idx + 1), 100. * correct_net3 / total,
                 test_loss_ensemble /
                 (batch_idx + 1), 100. * correct_ensemble / total))

    print('logdet %.2f || entropy %.2f' %
          (logdet / len(testloader), entropy / len(testloader)))

    # Save checkpoint.
    print('Saving..')
    state = {
        'net1': net1.state_dict(),
        'net2': net2.state_dict(),
        'net3': net3.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    ckpname = ('./checkpoint/ADP_' + args.name + '.pth')
    torch.save(state, ckpname)

    return (test_loss_net1 / (batch_idx + 1), test_loss_net2 / (batch_idx + 1),
            test_loss_net3 / (batch_idx + 1),
            test_loss_ensemble / (batch_idx + 1), 100. * correct_net1 / total,
            100. * correct_net2 / total, 100. * correct_net3 / total,
            100. * correct_ensemble / total)


def adjust_learning_rate(optimizer, epoch, lr_model):
    """decrease the learning rate"""
    lr = lr_model
    if epoch < 10:
        lr = lr * (epoch + 1) / 10
    if epoch > 100:
        lr = lr * .1
    if epoch > 150:
        lr = lr * .1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


trainloss_net1 = []
trainloss_net2 = []
trainloss_net3 = []
testloss_net1 = []
testloss_net2 = []
testloss_net3 = []
testloss_ensemble = []

trainacc_net1 = []
trainacc_net2 = []
trainacc_net3 = []
testacc_net1 = []
testacc_net2 = []
testacc_net3 = []
testacc_ensemble = []
#test(199)
print(args)
for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer_net1, epoch, args.lr)
    adjust_learning_rate(optimizer_net2, epoch, args.lr)
    adjust_learning_rate(optimizer_net3, epoch, args.lr)
    train_loss_net1, train_loss_net2, train_loss_net3,\
        train_acc_net1, train_acc_net2, train_acc_net3 = train(epoch)
    trainloss_net1.append(train_loss_net1)
    trainloss_net2.append(train_loss_net2)
    trainloss_net3.append(train_loss_net3)

    trainacc_net1.append(train_acc_net1)
    trainacc_net2.append(train_acc_net2)
    trainacc_net3.append(train_acc_net3)

    test_loss_net1, test_loss_net2, test_loss_net3, test_loss_ensemble,\
        test_acc_net1, test_acc_net2, test_acc_net3, test_acc_ensemble = test(epoch)
    testloss_net1.append(test_loss_net1)
    testloss_net2.append(test_loss_net2)
    testloss_net3.append(test_loss_net3)
    testloss_ensemble.append(test_loss_ensemble)

    testacc_net1.append(test_acc_net1)
    testacc_net2.append(test_acc_net2)
    testacc_net3.append(test_acc_net3)
    testacc_ensemble.append(test_acc_ensemble)

    plt.plot(trainloss_net1, 'b-', label='trainloss_net1 loss')
    plt.plot(trainloss_net2, 'g-', label='trainloss_net2 loss')
    plt.plot(trainloss_net3, 'r-', label='trainloss_net3 loss')
    plt.plot(testloss_net1, 'b--', label='testloss_net1 loss')
    plt.plot(testloss_net2, 'g--', label='testloss_net2 loss')
    plt.plot(testloss_net3, 'r--', label='testloss_net3 loss')
    plt.plot(testloss_ensemble, 'k--', label='testloss_ensemble loss')
    plt.legend(frameon=False)
    pltname = ('./plots/ADP_' + args.name + '_loss.png')
    plt.savefig(pltname)
    plt.close()

    plt.plot(trainacc_net1, 'b-', label='trainacc_net1 acc')
    plt.plot(trainacc_net2, 'g-', label='trainacc_net2 acc')
    plt.plot(trainacc_net3, 'r-', label='trainacc_net3 acc')
    plt.plot(testacc_net1, 'b--', label='testacc_net1 acc')
    plt.plot(testacc_net2, 'g--', label='testacc_net2 acc')
    plt.plot(testacc_net3, 'r--', label='testacc_net3 acc')
    plt.plot(testacc_ensemble, 'k--', label='testacc_ensemble acc')
    plt.legend(frameon=False)
    pltname = ('./plots/ADP_' + args.name + '_acc.png')
    plt.savefig(pltname)
    plt.close()
