from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import models
# from model_utils import *
# from datasets import *
import numpy as np
from collections import OrderedDict
from autoattack import AutoAttack

from sklearn.model_selection import train_test_split
from data_combine1021Alpha_great_combineAll import data_combine
from subDataset import subDataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.5,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='./2025_checkpoints/pgd_ResNet10_twin_adv_0.7_50_unstructure.pt',
                    help='model for white-box attack evaluation')
# parser.add_argument("--dataset", type=str, choices=["CIFAR10", "SVHN", "CIFAR100"], default="CIFAR10")
parser.add_argument("--model", type=str, choices=["ResNet18", "ResNet34", "ResNet50", "vgg16", 'WideResNet'], default="ResNet10")
parser.add_argument('--log-path', type=str, default='./log_file.txt')
parser.add_argument('--version', type=str, default='standard')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

'''
# set up data loader
if args.dataset == 'CIFAR10':
    train_loader, test_loader, num_class = cifar10_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
elif args.dataset == 'CIFAR100':
    train_loader, test_loader, num_class = cifar100_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
elif args.dataset == 'SVHN':
    train_loader, test_loader, num_class = svhn_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
elif args.dataset == 'tiny_imagenet':
    train_loader, test_loader, num_class = tiny_imagenet_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
elif args.dataset == 'imagenet':
    train_loader, test_loader, num_class = imagenet_dataloader(batch_size=args.batch_size)
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
'''

Data, Label, record, result = data_combine(False, False, 5, True, True, True)
X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.05, random_state=40)
train_xt = torch.from_numpy(X_train.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))
trainData = subDataset(train_xt, train_yt)
testData = subDataset(test_xt, test_yt)
train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=False)

def main():

    # cl, ll = get_layers('dense')
    # model = models.__dict__[args.model](conv_layer=cl, linear_layer=ll).to(device)
    # state_dict = torch.load(args.model_path)
    dict = torch.load('./1203_trained/best_base_model_ft1213_26.pth.tar')
    model = dict['model'].to(device)
    state_dict = dict['state_dict']
    # torch.save({'state_dict': state_dict, 'model': model},'./2025_checkpoints/pgd_ResNet10_twin_adv_0.7_50_unstructure.pth')

    new_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            new_k = k[7:]
        else:
            new_k = k
        new_dict[new_k] = v

    model.load_state_dict(new_dict)
    model.eval()

    adversary = AutoAttack(model, norm='Linf', eps=args.epsilon, log_path=args.log_path, version=args.version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    if args.version == 'custom':
        adversary.attacks_to_run = ['square']
        for n_queries in [100, 500]:
            adversary.square.n_queries = n_queries
            with torch.no_grad():
                adversary.run_standard_evaluation(x_test, y_test,
                    bs=args.test_batch_size)
    else:
        with torch.no_grad():
            adversary.run_standard_evaluation(x_test, y_test,
                bs=args.test_batch_size)


if __name__ == '__main__':
    main()
