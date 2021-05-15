from glob import glob
import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm
import argparse

import torch
torch.backends.cudnn.enabled = True
from torch.utils import data
import math
from scipy.ndimage.interpolation import zoom
import pdb
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

from dataset_effective_bsz import DatasetStg1
import monai
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
import os, re, time
import logging

set_determinism(seed=0) # set deterministic training for reproducibility

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="exp")
    parser.add_argument('--weights_file', type=str, default="best_weights")
    parser.add_argument('--activation', type=str, default="relu", help="[relu|leaky_relu]")
    parser.add_argument('--transform', type=str, default=None, help="[rand_affine|rand_elastic|rand_spatial_crop|rand_zoom]")
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--effective_bsz', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--background', dest='include_background', action='store_true')
    parser.add_argument('--no-background', dest='include_background', action='store_false')
    parser.set_defaults(include_background=False)
    parser.add_argument('--squared', dest='squared_pred', action='store_true')
    parser.add_argument('--no-squared', dest='squared_pred', action='store_false')
    parser.set_defaults(squared_pred=True)

    args = parser.parse_args()
    return args

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.match(pattern, f):
            os.remove(os.path.join(dir, f))

def main():
    args = get_arguments()
    print(args)

    TRAIN_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_crp_v2_pool1.pth'
    TEST_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/testpddca15_crp_v2_pool1.pth'
    CET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_cet_crp_v2_pool1.pth'
    PET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_pet_crp_v2_pool1.pth'
    if not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)
    purge(args.exp_dir, "events")
    writer = SummaryWriter(args.exp_dir)
    logging.basicConfig(filename=os.path.join(args.exp_dir, f'{args.exp_dir}.log'), filemode='w', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
    logging.info(args)
    weights_file = os.path.join(args.exp_dir, f"{args.weights_file}.pth")


    traindataset = DatasetStg1(PET_PATH, istranform=True, monai_transform=args.transform)
    traindataloader = torch.utils.data.DataLoader(traindataset,num_workers=0,batch_size=1, shuffle=True)
    testdataset = DatasetStg1(TEST_PATH, istranform=False)
    testdataloader = torch.utils.data.DataLoader(testdataset,num_workers=0,batch_size=1,shuffle=False)
    print(len(traindataloader), len(testdataloader))
    
    model = monai.networks.nets.BasicUNet(dimensions=3, in_channels=1, out_channels=10).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    print(f"Effective bsz = {args.effective_bsz}; effective lr = {args.lr}")
    logging.info(f"Effective bsz = {args.effective_bsz}; effective lr = {args.lr}")
    best_dc = -1
    best_per_class_dcs = [-1 for i in range(9)]

    start_time = time.time()
    

    # effective bsz links
    # 1. https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    # 2. https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    for epoch in range(args.num_epochs):
        epoch_time = time.time()
        trainloss = 0
        optimizer.zero_grad()
        for batch_idx, (x_train, y_train, flagvec) in enumerate(traindataloader):
            # goes OOM on this example
            if x_train.shape == torch.Size([1, 1, 168, 328, 232]):
                continue
            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.cuda())
            pred = model(x_train)
            pred = F.softmax(pred, dim=1)
            loss = monai.losses.DiceLoss(include_background=args.include_background, squared_pred=args.squared_pred)(pred, y_train)
            trainloss += loss.item()
            # we need to divide by effective bsz since usually we take mean along batch dim to get final loss
            # note that lr is also scaled according to effective bsz
            loss = loss/args.effective_bsz
            loss.backward()
            if (batch_idx+1)%args.effective_bsz == 0:
                optimizer.step()
                optimizer.zero_grad()

            
            # del loss, x_train, y_train, pred
        
        print(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Time = {round(time.time()-epoch_time, 5)}")
        logging.info(f"Epoch [{epoch}] Total Time = {round(time.time()-start_time, 5)}")
        writer.add_scalar('Loss/train', trainloss, epoch)
        
        if epoch % 10 == 0:
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
            
            best_per_class_dcs = [max(l1,l2) for l1,l2 in zip(test_per_class_dcs, best_per_class_dcs)]
            if best_dc < test_dc:
                best_dc = test_dc
                state = {"epoch": epoch, "best_dc": best_dc, "best_per_class_dcs": [i.item() for i in best_per_class_dcs], "weight": model.state_dict()}
                torch.save(state, weights_file)
            print(f"Epoch [{epoch}] Test DC = {test_dc}")
            print(f"Epoch [{epoch}] Test Per Class DCs = {[i.item() for i in best_per_class_dcs]}")
            logging.info(f"Epoch [{epoch}] Test DC = {test_dc}")
            logging.info(f"Epoch [{epoch}] Test Per Class DCs = {[i.item() for i in best_per_class_dcs]}")
            writer.add_scalar('Mean DC/test', test_dc, epoch)
            for i, cls_dc in enumerate(test_per_class_dcs):
                writer.add_scalar(f'DC/class {i}', cls_dc.item(), epoch)


if __name__ == "__main__":
    main()