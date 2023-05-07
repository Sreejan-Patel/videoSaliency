import torch
import torch.nn as nn
from loss import *
import cv2
from torchvision import transforms, utils
from PIL import Image

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        

def loss_func(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    assert pred_map.size() == gt.size()

    if len(pred_map.size()) == 4:
        assert pred_map.size(0)==args.batch_size
        pred_map = pred_map.permute((1,0,2,3))
        gt = gt.permute((1,0,2,3))

        for i in range(pred_map.size(0)):
            loss += get_loss(pred_map[i], gt[i], args)

        loss /= pred_map.size(0)
        return loss
    
    return get_loss(pred_map, gt, args)


def get_loss(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).cuda()
    loss += 1.0 * Loss.kldiv(pred_map, gt)

    return loss
