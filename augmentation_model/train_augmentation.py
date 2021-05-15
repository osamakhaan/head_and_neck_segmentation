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

from dataset_augmentation import DatasetStg1
from test_utils import test

set_determinism(seed=0) # set deterministic training for reproducibility

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="exp")
    parser.add_argument('--weights_file', type=str, default="best_weights")
    parser.add_argument('--model', type=str, default="unet", help="[unet|deeper_unet|segresnet]")
    parser.add_argument('--activation', type=str, default="relu", help="[relu|leaky_relu]")
    parser.add_argument('--transform', type=str, default=None, help="[rand_affine|rand_elastic|rand_spatial_crop|rand_zoom]")
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--background', dest='include_background', action='store_true')
    parser.add_argument('--no-background', dest='include_background', action='store_false')
    parser.set_defaults(include_background=False)
    parser.add_argument('--squared', dest='squared_pred', action='store_true')
    parser.add_argument('--no-squared', dest='squared_pred', action='store_false')
    parser.set_defaults(squared_pred=True)
    parser.add_argument('--delay', dest='delay_transform', action='store_true')
    parser.add_argument('--no-delay', dest='delay_transform', action='store_false')
    parser.set_defaults(delay_transform=False)
    parser.add_argument('--weight', dest='weight_difficult_classes', action='store_true')
    parser.add_argument('--no-weight', dest='weight_difficult_classes', action='store_false')
    parser.set_defaults(weight_difficult_classes=False)

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


    transform_traindataset = DatasetStg1(PET_PATH, istranform=True, monai_transform=args.transform)
    transform_traindataloader = torch.utils.data.DataLoader(transform_traindataset,num_workers=0,batch_size=1, shuffle=True)
    if args.delay_transform:
        traindataset = DatasetStg1(PET_PATH, istranform=True, monai_transform=None)
        traindataloader = torch.utils.data.DataLoader(traindataset,num_workers=0,batch_size=1, shuffle=True)
        print("Using normal data for first 40 epochs followed by transformed data!")
        logging.info("Using normal data for first 40 epochs followed by transformed data!")
    testdataset = DatasetStg1(TEST_PATH, istranform=False)
    testdataloader = torch.utils.data.DataLoader(testdataset,num_workers=0,batch_size=1,shuffle=False)
    print(len(transform_traindataloader), len(testdataloader))
    
    if args.model == "deeper_unet":
        logging.info("Training with deeper UNet model!")
        print("Training with deeper UNet model!")
        model = monai.networks.nets.BasicUNet(dimensions=3, in_channels=1, out_channels=10, features=(32, 64, 128, 256, 512, 32)).cuda()
    elif args.model == "unet":
        logging.info("Training with normal UNet model!")
        print("Training with normal UNet model!")
        model = monai.networks.nets.BasicUNet(dimensions=3, in_channels=1, out_channels=10).cuda()
    elif args.model == "segresnet":
        logging.info("Training with SegResNet model!")
        print("Training with SegResNet model!")
        model = monai.networks.nets.SegResNet(spatial_dims=3, in_channels=1, out_channels=10, upsample_mode="deconv").cuda()
    else:
        raise Exception("Invalid model!")
    optimizer = torch.optim.RMSprop(model.parameters(),lr = args.lr)
    best_dc = -1
    best_per_class_dcs = [-1 for i in range(9)]

    start_time = time.time()
    for epoch in range(args.num_epochs):
        epoch_time = time.time()
        trainloss = 0
        if args.delay_transform and epoch < 40:
            loader = traindataloader
        else:
            loader = transform_traindataloader
        for x_train, y_train, flagvec in loader:
            # goes OOM on this example
            if x_train.shape == torch.Size([1, 1, 168, 328, 232]):
                continue
            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.cuda())
            optimizer.zero_grad()
            pred = model(x_train)
            pred = F.softmax(pred, dim=1)
            if args.weight_difficult_classes:
                loss = monai.losses.DiceLoss(include_background=args.include_background, squared_pred=args.squared_pred, reduction="none")(pred, y_train)
                weights = [1.0]*loss.shape[-1]
                for k in range(loss.shape[-1]):
                    if loss[0,k] >= 0.95:
                        weights[k] = 5.0
                loss = (loss*torch.FloatTensor(weights).cuda()).mean()
            else:
                loss = monai.losses.DiceLoss(include_background=args.include_background, squared_pred=args.squared_pred)(pred, y_train)
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
            del loss, x_train, y_train, pred
        
        print(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Time = {round(time.time()-epoch_time, 5)}")
        logging.info(f"Epoch [{epoch}] Total Time = {round(time.time()-start_time, 5)}")
        writer.add_scalar('Loss/train', trainloss, epoch)
        
        if epoch % 10 == 0:
            test_dc, test_per_class_dcs = test(testdataloader, args, epoch, writer, model)
            
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