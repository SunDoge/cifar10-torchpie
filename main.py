# Follow https://github.com/pytorch/examples/blob/master/imagenet/main.py

import time

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as T
# import utils.transforms as UT
# from datasets.cifar import CIFAR10
from torchvision.datasets import CIFAR10

import torchpie.parallel as tpp
from models import get_model
from torchpie.config import config
from torchpie.environment import experiment_path, args
from torchpie.logging import logger
from torchpie.meters import AverageMeter
from torchpie.parallel.utils import reduce_tensor, scale_lr
from torchpie.utils.checkpoint import save_checkpoint


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


# @profile
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
        data_time.update(time.perf_counter() - end)

        images, target = images.cuda(
            non_blocking=True), target.cuda(non_blocking=True)

        output = model(images)

        loss = criterion(output, target)

        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

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

    return top1.avg


def validate(model: nn.Module, loader, criterion, epoch):
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
                non_blocking=True), target.cuda(non_blocking=True)

            output = model(images)

            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if tpp.distributed:
                loss = reduce_tensor(loss)
                acc1 = reduce_tensor(acc1)
                acc5 = reduce_tensor(acc5)

            batch_size = target.shape[0]
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            logger.info(
                f'Epoch [{epoch}][{i}/{loader_len}]\t'
                f'{batch_time}\t{losses}\t{top1}\t{top5}'
            )

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc1', top1.avg, epoch)
    writer.add_scalar('val/acc5', top5.avg, epoch)

    return top1.avg


def main():
    global best_acc1, start_epoch
    model = get_model(config.get_string('arch'))

    model.cuda()

    learning_rate = scale_lr(
        config.get_float('optimizer.lr'),
        config.get_int('dataloader.batch_size')
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config.get_float('optimizer.momentum'),
        weight_decay=config.get_float('optimizer.weight_decay'),
        nesterov=config.get_bool('optimizer.nesterov')
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        config.get_list('scheduler.milestones')
    )

    if tpp.distributed:
        model = DistributedDataParallel(model, device_ids=[tpp.local_rank])

    normalize = T.Normalize(
        config.get_list('dataset.mean'),
        config.get_list('dataset.std')
    )
    train_transform = T.Compose([
        # UT.RandomCrop(32, padding=4),
        # UT.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        normalize
    ])

    train_set = CIFAR10(
        config.get_string('dataset.root'), train=True, transform=train_transform, download=True
    )
    val_set = CIFAR10(
        config.get_string('dataset.root'), train=False, transform=val_transform, download=False
    )

    train_sampler = None
    val_sampler = None
    if tpp.distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=config.get_int('dataloader.batch_size'),
        pin_memory=True,
        shuffle=(train_sampler is None),
        num_workers=config.get_int('dataloader.num_workers'),
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.get_int('dataloader.batch_size'),
        pin_memory=True,
        num_workers=config.get_int('dataloader.num_workers'),
        sampler=val_sampler
    )

    for epoch in range(start_epoch, config.get_int('strategy.num_epochs')):
        # for epoch in range(start_epoch, 1):

        if tpp.distributed:
            train_sampler.set_epoch(epoch)

        train(model, train_loader, criterion, optimizer, epoch)
        acc1 = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config.get_string('arch'),
            'state_dict': model.module.state_dict() if tpp.distributed else model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=is_best, folder=experiment_path)


if __name__ == "__main__":

    if experiment_path is not None:
        # Create it when local_rank == 0
        writer = tpp.rank0(lambda: SummaryWriter(log_dir=experiment_path))()

    best_acc1 = 0.0
    start_epoch = 0

    main()
    writer.close()
