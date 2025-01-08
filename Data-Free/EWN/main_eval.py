import argparse
import copy
import shutil
import time
import warnings
from collections import OrderedDict
from enum import Enum

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from ptflops import get_model_complexity_info
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_combine1021Alpha_great_combineAll import data_combine
from loss_function_0712Alpha import loss_soft_add
from loss_function_0712Alpha import test_soft_add
# from pruner.genthin_resnet10 import GenThinPruner
from subDataset import subDataset
from attack import pgd_attack
# from try_resnet_1003 import ResNet, BasicBlock
# from try_resnet_0706 import ResNet, BasicBlock
# from models import *
# from resnet_mqa import ResNet10

parser = argparse.ArgumentParser(description='Model Pruning Implementation')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--accuracy-threshold', default=2.5, type=float,
                    help='validation accuracy drop feasible for the pruned model', dest='accuracy_threshold')
parser.add_argument('--pruning-percentage', default=0.01, type=float,
                    help='percentage of channels to prune per pruning iteration', dest='pruning_percentage')
parser.add_argument('--num-processes', default=3, type=int,
                    help='number of simultaneous process to spawn for multiprocessing', dest='num_processors')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--scoring-strategy', default='dfpc', type=str, help='strategy to compute saliencies of channels',
                    dest='strategy', choices=['dfpc', 'l1', 'random'])
parser.add_argument('--prune-coupled', default=1, type=int, help='prune coupled channels is set to 1',
                    dest='prunecoupled', choices=[0, 1])
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

args = parser.parse_args()


def main():
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    Data, Label, record, result = data_combine(False, False, 5, True, True, True)
    X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.05, random_state=40)
    train_xt = torch.from_numpy(X_train.astype(np.float32))
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt = torch.from_numpy(X_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))
    trainData = subDataset(train_xt, train_yt)
    testData = subDataset(test_xt, test_yt)
    train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers)

    # Load checkpoint and define model
    pruning_iteration = 0
    print("=> using pre-trained model")
    dict = torch.load('./1203_trained/best_base_model_ft1213_26.pth.tar')
    base_model = dict['model']
    base_model.load_state_dict(dict['state_dict'])

    net = model = copy.deepcopy(base_model)
    macs, params = get_model_complexity_info(net, (10, 5, 5), as_strings=True, print_per_layer_stat=False)
    del net
    model = ToAppropriateDevice(copy.deepcopy(base_model), args)

    cudnn.benchmark = True

    # define loss function (criterion)
    # criterionnew = nn.L1Loss().cuda(args.gpu)
    criterion = loss_soft_add().cuda(args.gpu)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                  amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-4)

    accuracy = validate(val_loader, model, criterion, args)
    print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
    unpruned_accuracy = accuracy

    # evaluation on natural examples
    print('================================================================')
    # train_loss, train_loss_pgd, train_acc, train_acc_pgd = eval_train(model, device, train_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculate_sparsity(model)
    test_loss, test_loss_pgd, test_acc, test_acc_pgd = eval_test(model, device, val_loader)
    print('================================================================')

    '''
    # print('Initializing Pruner...')
    # pruner = GenThinPruner(base_model, args)
    # print('Computing Saliency Scores...')
    # pruner.ComputeSaliencyScores(base_model)
    while accuracy > 7.5 and pruning_iteration < 50:
        pruning_iteration += 1
        print('Pruning iteration {}...'.format(pruning_iteration))
        # print('Pruning the model...')
        # num_channels_to_prune = int(args.pruning_percentage * pruner.total_channels(base_model))
        # for _ in range(num_channels_to_prune):
        #     pruner.Prune(base_model)

        model = copy.deepcopy(base_model)
        model = torch.nn.DataParallel(model).cuda()

        acc1 = finetune(model, train_loader, val_loader, criterion, optimizer, scheduler)
        # acc1 = validate(val_loader, model, criterion, args)

        # remember best pruned model and save checkpoint
        is_best = (acc1 <= accuracy - args.accuracy_threshold)

        save_checkpoint({
            'pruning_iteration': pruning_iteration,
            'unpruned_accuracy': unpruned_accuracy,
            'model': model,
            'state_dict': model.state_dict(),
            'acc1': acc1,
        }, is_best, filename='dataparallel_model_test1213.pth.tar')

        save_checkpoint({
            'model': base_model,
        }, is_best, filename='base_model_test1213.pth.tar')

        _save_checkpoint({
            'model': base_model,
            'state_dict': base_model.state_dict(),
        }, is_best, filename='base_model_test1213_' + str(pruning_iteration) + '.pth.tar')

        del model, base_model

        base_model, _, _ = LoadBaseModel()
        net = model = copy.deepcopy(base_model)
        macs, params = get_model_complexity_info(net, (10, 5, 5), as_strings=True, print_per_layer_stat=False)
        del net
        print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
        accuracy = acc1
    '''


def finetune(model, train_loader, val_loader, criterion, optimizer, scheduler):
    global best_acc1
    for epoch in range(args.epochs):
        print(
            'Finetuning epoch {} of {} with learning rate {}.'.format(epoch + 1, args.epochs, scheduler.get_last_lr()))

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        scheduler.step()
    return acc1


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        test_cal = test_soft_add().cuda(args.gpu)
        acc1 = test_cal(output, target)
        acc1 = 100 * acc1 ** 0.5
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            test_cal = test_soft_add().cuda(args.gpu)
            acc1 = test_cal(output, target)
            acc1 = 100 * acc1 ** 0.5
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_' + filename)

def _save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    _filename = './1213_trained/' + filename
    torch.save(state, _filename)
    if is_best:
        shutil.copyfile(_filename, './1213_trained/best_' + filename)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush = True)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries), flush = True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


'''
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
'''


def DataParallelStateDict_To_StateDict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def ToAppropriateDevice(model, args):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model

def LoadBaseModel():
    base_model_dict = torch.load('base_model_test1213.pth.tar', map_location=torch.device('cpu'))
    base_model = base_model_dict['model']
    model_dict = torch.load('dataparallel_model_test1213.pth.tar', map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    unpruned_accuracy = model_dict['unpruned_accuracy']
    pruning_iteration = model_dict['pruning_iteration']
    base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
    del base_model_dict, model_dict, state_dict
    base_model = base_model.to('cpu')
    return base_model, unpruned_accuracy, pruning_iteration

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_loss_pgd = 0
    correct = 0
    correct_pgd = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            loss, loss_pgd, acc, acc_pgd = pgd_attack(model, data, target, device=device)
            correct += acc
            correct_pgd += acc_pgd
            test_loss += loss
            test_loss_pgd += loss_pgd
    test_loader_len = len(test_loader)
    test_loss /= test_loader_len
    test_loss_pgd /= test_loader_len
    test_accuracy = 100. * correct / test_loader_len
    test_accuracy_pgd = 100. * correct_pgd / test_loader_len
    print('Test: Average loss: {:.4f}, Accuracy: {:.4f}%'.format(
        test_loss, test_accuracy))
    print('Test: Average loss: {:.4f}, Accuracy: {:.4f}%'.format(
        test_loss_pgd, test_accuracy_pgd))
    return test_loss, test_loss_pgd, test_accuracy, test_accuracy_pgd


def calculate_sparsity(model):
    total_params = 0
    zero_params = 0
    layer_sparsity = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            # 计算当前层的总参数量和零参数量
            layer_total = param.numel()
            layer_zero = (param == 0).sum().item()

            # 更新总参数量和零参数量
            total_params += layer_total
            zero_params += layer_zero

            # 计算当前层稀疏度并记录
            layer_sparsity[name] = layer_zero / layer_total

    # 计算整体模型稀疏度
    sparsity = zero_params / total_params

    # 打印统计结果
    print(f"Model Sparsity Analysis:")
    print(f"Total parameters: {total_params}")
    print(f"Zero parameters: {zero_params}")
    print(f"Overall Sparsity: {sparsity:.2%}\n")

    print("Layer-wise Sparsity:")
    for layer_name, layer_spar in layer_sparsity.items():
        print(f"  {layer_name}: {layer_spar:.2%}")


if __name__ == '__main__':
    main()
