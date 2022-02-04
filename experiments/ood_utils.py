"""OOD helpers

Reference: https://github.com/eminorhan/ood-benchmarks/blob/master/utils.py
"""

import os
import time
from black import out
import torch
import torchvision

from experiments.experiment_utils import REPO_PATH


class ShortEdgeCenterCrop():
    """Center crop for tf models"""
    def __call__(self, x):
        im_width, im_height = x.size
        padded_center_crop_size = min(im_height, im_width)

        return torchvision.transforms.functional.center_crop(x, padded_center_crop_size)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def simple_validate(val_loader, model, gpu=None, print_freq=10):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                # images = images.cuda(gpu, non_blocking=True)
                # target = target.cuda(gpu, non_blocking=True)
                images=images.to(gpu)
                target=target.to(gpu)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def imagenet_adv_validate(val_loader, model, gpu=None, print_freq=10, eval_indices=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                # images = images.cuda(gpu, non_blocking=True)
                # target = target.cuda(gpu, non_blocking=True)
                images=images.to(gpu)
                target=target.to(gpu)

            # compute output
            output = model(images)
            output = output[:,eval_indices]

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def stylized_imagenet_validate(val_loader, model, gpu=None, print_freq=10):
    import sin_utils 
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                # images = images.cuda(gpu, non_blocking=True)
                # target = target.cuda(gpu, non_blocking=True)
                images=images.to(gpu)
                target=target.to(gpu)

            # compute output
            output = model(images)
            mapping = sin_utils.ImageNetProbabilitiesTo16ClassesMapping()

            import helper.human_categories as hc
            c = hc.HumanCategories()

            output = output.softmax(1).cpu().numpy()
            output = [mapping.probabilities_to_decision(x) for x in output]
            output = [hc.get_human_object_recognition_categories().index(x) for x in output]
            # output = torch.tensor(output)
            import numpy as np
            acc1 = np.mean(np.array(output) == target.cpu().numpy()) * 100

            # print(output, target, c.get_imagenet_indices_for_category(output[0]))
            # print()
            # output = torch.stack(output)

            # measure accuracy and record loss
            # acc1 = accuracy(output, target, topk=(1, ))
            # top1.update(acc1[0].cpu().numpy()[0], images.size(0))
            top1.update(acc1, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg


def load_model(model_name):
    "Loads one of the pretrained models."

    torch_hub_dir = '/misc/vlgscratch4/LakeGroup/emin/robust_vision/pretrained_models'
    torch.hub.set_dir(torch_hub_dir)

    if model_name.startswith('resnext101'):
        if model_name == 'resnext101_32x8d':
            model = torchvision.models.resnext101_32x8d(pretrained=True)
        else:
            model = torch.hub.load('facebookresearch/WSL-Images', model_name)
        model = torch.nn.DataParallel(model).cuda()
    elif model_name.startswith('tf_efficientnet'):
        model = torch.hub.load('rwightman/gen-efficientnet-pytorch', model_name, pretrained=True)
        model = torch.nn.DataParallel(model).cuda()
    elif model_name == 'moco_v2':
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(os.path.join(torch_hub_dir, 'moco_v2.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.DataParallel(model).cuda()
    else:
        raise ValueError('Model not available.')

    print('Loaded model:', model_name)

    return model