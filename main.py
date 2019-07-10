# Follow https://github.com/pytorch/examples/blob/master/imagenet/main.py

from torchpie.config import config
import torchpie.parallel as tpp
from torchpie.parallel.reducer import reduce_tensor
from torchpie.parallel.scaler import scale_lr
from torch.utils.tensorboard import SummaryWriter
from torchpie.environment import experiment_path
from torchpie.logging import logger
from torch import nn, optim
import time
import torch
from models import get_model
from torch.nn.parallel import DistributedDataParallel


class AverageMeter:

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model: nn.Module, loader, criterion, optimzier, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()

    end = time.perf_counter()

    loader_len = len(loader)

    for i, (images, target) in enumerate(loader):
        images, target = images.cuda(
            non_blocking=True), target.cuda(non_bloking=True)

        output = model(images)

        loss = criterion(output, target)

        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if tpp.distributed:
            loss = reduce_tensor(loss)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)

        batch_size = target.shape[0]
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        logger.info(
            f'Epoch [{epoch}][{i}/{loader_len}]\t'
            f'{batch_time}\t{data_time}\t{losses}\t{top1}\t{top5}'
        )

    writer.add_scalar('train/loss', loss.item(), epoch)
    writer.add_scalar('train/acc1', top1.avg, epoch)
    writer.add_scalar('train/acc5', top5.avg, epoch)


def validate(model: nn.Module, loader, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    loader_len = len(loader)

    model.eval()

    with torch.no_grad():
        end = time.perf_counter()

        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(
                non_blocking=True), target.cuda(non_bloking=True)

            output = model(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if tpp.distributed:
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)

            batch_size = target.shape[0]
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            logger.info(
                f'Epoch [{epoch}][{i}/{loader_len}]\t'
                f'{batch_time}\t{losses}\t{top1}\t{top5}'
            )

    writer.add_scalar('train/acc1', top1.avg, epoch)
    writer.add_scalar('train/acc5', top5.avg, epoch)

    return top1.avg


def main():

    model = get_model(config.get_string('arch'))

    optimizer = optim.SGD()
    criterion = nn.CrossEntropyLoss()

    if tpp.distributed:
        model = DistributedDataParallel(model, device_ids=[tpp.local_rank])

    


if __name__ == "__main__":

    if experiment_path is not None:
        writer = tpp.rank0_wrapper(SummaryWriter(
            log_dir=experiment_path), SummaryWriter)

    main()
