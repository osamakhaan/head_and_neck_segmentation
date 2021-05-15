# Python imports
import os, math, pdb, re, time
from glob import glob
from tqdm import tqdm
import argparse
import numpy as np
import logging

# ML imports
from scipy.ndimage.interpolation import zoom
from scipy.spatial.distance import dice
import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# Medical imaging specific imports
import SimpleITK as sitk
import monai
from monai.utils import set_determinism

def test(testdataloader, args, epoch, writer, model):
    test_dc = 0.0
    test_per_class_dcs = torch.FloatTensor([0.0 for i in range(9)])
    for idx, (x, y, _) in enumerate(testdataloader):
        with torch.no_grad():
            x = torch.autograd.Variable(x.cuda())
            pred = model(x)
            pred = F.softmax(pred, dim=1)
            # convert pred to one hot
            max_idx = torch.argmax(pred.cpu(),dim=1,keepdim=True)
            one_hot_pred = torch.FloatTensor(pred.shape)
            one_hot_pred.zero_()
            one_hot_pred.scatter_(1,max_idx,1)

            tmp = monai.metrics.compute_meandice(one_hot_pred, y.cpu(), include_background=args.include_background)
            dc, per_class_dcs = tmp.mean(), tmp.mean(dim=0)
            test_dc += dc.item()
            test_per_class_dcs += per_class_dcs
            n_classes = 10 if args.include_background else 9
            for l in range(n_classes):
                if n_classes == 9:
                    m = l+1
                else:
                    m = l
                # gt
                monai.visualize.img2tensorboard.add_animated_gif(writer, f"Ex {idx}/Label {m}", y[:,m,:,:,:].permute(0,2,3,1).cpu(), max_out=x.shape[1], scale_factor=255, global_step=epoch)
                # pred
                monai.visualize.img2tensorboard.add_animated_gif(writer, f"Ex {idx}/Label {m}/DC", one_hot_pred[:,m,:,:,:].permute(0,2,3,1).cpu(), max_out=x.shape[1], scale_factor=255, global_step=epoch)
            # image
            monai.visualize.img2tensorboard.add_animated_gif(writer, f"Image {idx}/DC {round(dc.item(),3)}", x[0].permute(0,2,3,1).cpu(), max_out=x.shape[1], scale_factor=255, global_step=epoch)

    test_dc = test_dc/len(testdataloader)
    test_per_class_dcs /= len(testdataloader)
    
    return test_dc, test_per_class_dcs