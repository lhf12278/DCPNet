import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from .model_upnet import UPNet
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # lms = kwargs.get('lms')
        # outputs = outputs + lms  # outputs: hp_sr
        # Compute all the requested losses
        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts

from UDL.pansharpening.models import PanSharpeningModel
class build_upnet(PanSharpeningModel, name='UPNet'):
    def __call__(self, args):
        scheduler = None
        if any(["wv" in v for v in args.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4

        loss = nn.L1Loss(size_average=True).cuda()
        # loss2 = nn.MSELoss(size_average=True).cuda()
        # loss = L1_Charbonnier_loss().cuda()
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        # weight_dict = {'loss1': 1,'loss2': 2}
        # losses = {'loss1': loss1, 'loss2': loss2}
        criterion = SetCriterion(losses, weight_dict)
        model = UPNet(spectral_num, criterion).cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)  ## optimizer 1: Adam

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=50,
        #                                                gamma=0.7)  # <=> lr = opt.lr * (0.5 ** (epoch // opt.step))

        return model, criterion, optimizer, scheduler