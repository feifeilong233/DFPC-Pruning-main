import os
import time
from tqdm import tqdm
from enum import Enum
from collections import OrderedDict

import copy
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info
from models import *


def main():
    # 硬编码参数
    data_path = './data'
    batch_size = 32
    workers = 4
    print_freq = 500
    checkpoint_path = 'best_dataparallel_model.pth.tar'

    print("=> using pre-trained model of 'resnet50'")
    base_model = ResNet50()
    state_dict = torch.load('./pretrained_checkpoints/cifar10_resnet50_ckpt.pth')['net']
    base_model.load_state_dict(DataParallelStateDict_To_StateDict(state_dict))
    del state_dict

    net = model = copy.deepcopy(base_model)
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    del net

    print('-{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('+{:<30}  {:<8}'.format('Number of parameters: ', params))


    # 加载模型检查点
    dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = dict['model']
    model.load_state_dict(dict['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_path, train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                         ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # 输出计算复杂度和参数数量
    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    print(f'-Computational complexity: {macs}')
    print(f'+Number of parameters: {params}')

    validate(val_loader, model, criterion, print_freq)


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 切换到评估模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # 计算输出
            output = model(images)
            loss = criterion(output, target)

            # 计算准确率并记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 计算消耗时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


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
        print('\t'.join(entries), flush=True)

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries), flush=True)

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
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    main()
