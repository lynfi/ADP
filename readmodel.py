'''Read models.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from normal import *


def cudamodel(net, device):
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    #return net


def loadmodel(net, name):
    ckpname = ('./checkpoint/' + '_' + name + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint['net'])


def loaddivmodel(net, name, ckpname):
    checkpoint = torch.load('./checkpoint/' + ckpname + '.pth')
    net.load_state_dict(checkpoint[name])