import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys,os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import to_one_hot, mixup_process, get_lambda
from load_data import per_image_standardization
import random

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_channels, num_classes,  per_img_std= False, stride=1):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        self.per_img_std = per_img_std
        #import pdb; pdb.set_trace()
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer1 = self._make_layer(block, initial_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, initial_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, initial_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, initial_channels*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(initial_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self,x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, target= None, mixup=False, mixup_hidden=False, mixup_alpha=None, loss_batch=None, in_batch=False, p=1.0,
                emd=False, proximal=True, reg=1e-5, itermax=10, label_inter=False, mean=None, std=None,
                box=False, graph=False, method='random', grad=None, block_num=32, beta=0.0, gamma=0., eta=0.2, neigh_size=2, n_labels=2, label_cost='l2',sigma=1.0, warp=0.0, dim=2, beta_c=0.0,
                transport=False, t_eps=10.0, t_type='full', t_size=16, noise=None, adv_mask1=0, adv_mask2=0):
        #import pdb; pdb.set_trace()
        if self.per_img_std:
            x = per_image_standardization(x)
        
        if mixup_hidden:
            layer_mix = random.randint(0,2)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None   
        
        out = x
        
        if target is not None :
            target_reweighted = to_one_hot(target,self.num_classes)
        
        if layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, mixup_alpha=mixup_alpha, loss_batch=loss_batch, p=p, in_batch=in_batch, 
                    emd=emd, proximal=proximal, reg=reg, itermax=itermax, label_inter=label_inter, mean=mean, std=std,
                    box=box, graph=graph, method=method, grad=grad, block_num=block_num,
                    beta=beta, gamma=gamma, eta=eta, neigh_size=neigh_size, n_labels=n_labels, label_cost=label_cost, sigma=sigma, warp=warp, dim=dim, beta_c=beta_c,
                    transport=transport, t_eps=t_eps, t_type=t_type, t_size=t_size, noise=noise, adv_mask1=adv_mask1, adv_mask2=adv_mask2)
            
        out = self.conv1(out)
        out = self.layer1(out)

        if layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, mixup_alpha=mixup_alpha, loss_batch=loss_batch, p=p, in_batch=in_batch, hidden=True,
                    emd=emd, proximal=proximal, reg=reg, itermax=itermax, label_inter=label_inter, mean=mean, std=std,
                    box=box, graph=graph, method=method, grad=grad, block_num=block_num, beta=beta, gamma=gamma, neigh_size=neigh_size, n_labels=n_labels)

        out = self.layer2(out)
        if layer_mix == 2:
             out, target_reweighted = mixup_process(out, target_reweighted, mixup_alpha=mixup_alpha, loss_batch=loss_batch, p=p, in_batch=in_batch, hidden=True,
                    emd=emd, proximal=proximal, reg=reg, itermax=itermax, label_inter=label_inter, mean=mean, std=std,
                    box=box, graph=graph, method=method, grad=grad, block_num=block_num, beta=beta, gamma=gamma, neigh_size=neigh_size, n_labels=n_labels)

        out = self.layer3(out)
        if  layer_mix == 3:
            out, target_reweighted = mixup_process(out, target_reweighted, mixup_alpha=mixup_alpha, loss_batch=loss_batch, p=p, in_batch=in_batch, hidden=True,
                    emd=emd, proximal=proximal, reg=reg, itermax=itermax, label_inter=label_inter, mean=mean, std=std,
                    box=box, graph=graph, method=method, grad=grad, block_num=block_num, beta=beta, gamma=gamma, neigh_size=neigh_size, n_labels=n_labels)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        if target is not None:
            return out, target_reweighted
        else: 
            return out


def preactresnet18(num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(PreActBlock, [2,2,2,2], 64, num_classes,  per_img_std, stride= stride)

def preactresnet34(num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(PreActBlock, [3,4,6,3], 64, num_classes,  per_img_std, stride= stride)

def preactresnet50(num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(PreActBottleneck, [3,4,6,3], 64, num_classes,  per_img_std, stride= stride)

def preactresnet101(num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(PreActBottleneck, [3,4,23,3], 64, num_classes, per_img_std, stride= stride)

def preactresnet152(num_classes=10, dropout = False,  per_img_std = False, stride=1):
    return PreActResNet(PreActBottleneck, [3,8,36,3], 64, num_classes, per_img_std, stride= stride)

def test():
    net = PreActResNet152(True,10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

if __name__ == "__main__":
    test()
# test()

