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
from pruner.genthin_single import GenThinPruner
from subDataset import subDataset
# from try_resnet_1003 import ResNet, BasicBlock
from try_resnet_0706 import ResNet, BasicBlock
# from models import *

parser = argparse.ArgumentParser(description='Model Pruning Implementation')
parser.add_argument('--resume', default='0830_1111_Alpha1.pt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--accuracy-threshold', default=2.5, type=float,
                    help='validation accuracy drop feasible for the pruned model', dest='accuracy_threshold')
parser.add_argument('--pruning-percentage', default=0.01, type=float,
                    help='percentage of channels to prune per pruning iteration', dest='pruning_percentage')
parser.add_argument('--num-processes', default=5, type=int,
                    help='number of simultaneous process to spawn for multiprocessing', dest='num_processors')
parser.add_argument('--scoring-strategy', default='dfpc', type=str, help='strategy to compute saliencies of channels',
                    dest='strategy', choices=['dfpc', 'l1', 'random'])
parser.add_argument('--prune-coupled', default=1, type=int, help='prune coupled channels is set to 1',
                    dest='prunecoupled', choices=[0, 1])

args = parser.parse_args()


def main():
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    Data, Label, record, result = data_combine(False, False, 5, True, True, True)
    X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.01, random_state=40)
    test_xt = torch.from_numpy(X_test.astype(np.float32))
    test_yt = torch.from_numpy(y_test.astype(np.float32))
    testData = subDataset(test_xt, test_yt)
    val_loader = torch.utils.data.DataLoader(dataset=testData, batch_size=64, shuffle=True)

    # Load checkpoint and define model
    pruning_iteration = 0
    print("=> using pre-trained model")
    base_model = ResNet(BasicBlock, [1, 1, 1, 1])
    base_model.load_state_dict(torch.load('./0830_1111_Alpha1.pt'))

    net = model = copy.deepcopy(base_model)
    macs, params = get_model_complexity_info(net, (10, 5, 5), as_strings=True, print_per_layer_stat=False)
    del net
    model = ToAppropriateDevice(copy.deepcopy(base_model), args)

    cudnn.benchmark = True

    # define loss function (criterion)
    criterion = loss_soft_add().cuda(args.gpu)

    accuracy = validate(val_loader, model, criterion, args)
    print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
    unpruned_accuracy = accuracy

    print('Initializing Pruner...')
    pruner = GenThinPruner(base_model, args)
    print('Computing Saliency Scores...')
    pruner.ComputeSaliencyScores(base_model)
    while accuracy >= 2.5:
        pruning_iteration += 1
        print('Pruning iteration {}...'.format(pruning_iteration))
        print('Pruning the model...')
        num_channels_to_prune = int(args.pruning_percentage * pruner.total_channels(base_model))
        for _ in range(num_channels_to_prune):
            pruner.Prune(base_model)

        model = copy.deepcopy(base_model)
        model = torch.nn.DataParallel(model).cuda()

        acc1 = validate(val_loader, model, criterion, args)

        # remember best pruned model and save checkpoint
        is_best = (acc1 >= unpruned_accuracy - args.accuracy_threshold)

        save_checkpoint({
            'pruning_iteration': pruning_iteration,
            'unpruned_accuracy': unpruned_accuracy,
            'model': model,
            'state_dict': model.state_dict(),
            'acc1': acc1,
        }, is_best, filename='dataparallel_model.pth.tar')

        save_checkpoint({
            'arch': args.arch,
            'model': base_model
        }, is_best, filename='base_model.pth.tar')

        del model, base_model

        base_model, _, _ = LoadBaseModel()
        net = model = copy.deepcopy(base_model)
        macs, params = get_model_complexity_info(net, (10, 5, 5), as_strings=True, print_per_layer_stat=False)
        del net
        print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('+{:<30}  {:<8}'.format('Number of parameters: ', params))
        accuracy = acc1


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
    base_model_dict = torch.load('base_model.pth.tar', map_location=torch.device('cpu'))
    base_model = base_model_dict['model']
    model_dict = torch.load('dataparallel_model.pth.tar', map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    unpruned_accuracy = model_dict['unpruned_accuracy']
    pruning_iteration = model_dict['pruning_iteration']
    base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
    del base_model_dict, model_dict, state_dict
    base_model = base_model.to('cpu')
    return base_model, unpruned_accuracy, pruning_iteration

if __name__ == '__main__':
    main()
