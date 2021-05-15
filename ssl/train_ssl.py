import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unet3d

from glob import glob
import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm
import argparse

torch.backends.cudnn.enabled = True
from torch.utils import data
import math
from scipy.ndimage.interpolation import zoom
import pdb
from torch import nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

from dataset_ssl import DatasetStg1
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
    parser.add_argument('--transform', type=str, default="crop1", help="[crop1|crop2]")
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--background', dest='include_background', action='store_true')
    parser.add_argument('--no-background', dest='include_background', action='store_false')
    parser.set_defaults(include_background=False)
    parser.add_argument('--squared', dest='squared_pred', action='store_true')
    parser.add_argument('--no-squared', dest='squared_pred', action='store_false')
    parser.set_defaults(squared_pred=True)
    parser.add_argument('--pretrain', action='store_true', default=False)

    args = parser.parse_args()
    return args

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.match(pattern, f):
            os.remove(os.path.join(dir, f))



# #Declare the Dice Loss
# def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
#     y_true_f = torch.flatten(y_true)
#     y_pred_f = torch.flatten(y_pred)
#     intersection = torch.sum(y_true_f * y_pred_f)
#     return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))


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
    testdataset = DatasetStg1(TEST_PATH, istranform=False, monai_transform=args.transform)
    testdataloader = torch.utils.data.DataLoader(testdataset,num_workers=0,batch_size=1,shuffle=False)
    print(len(traindataloader), len(testdataloader))

    # prepare the 3D model

    model = unet3d.UNet3D()

    if args.pretrain:
        #Load pre-trained weights
        weight_dir = '/scratch/mok232/Medical-Imaging/ModelsGenesis/pretrained_weights/Genesis_Chest_CT.pt'
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        model.load_state_dict(unParalled_state_dict)
        print("loaded pretrained weights")
        logging.info("loaded pretrained weights")
    else:
        print("training from scratch")
        logging.info("training from scratch") 

    # model.down_tr64.ops[0].conv1 = nn.Conv3d(4, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    model.out_tr.final_conv = nn.Conv3d(64, 10, kernel_size=(1, 1, 1), stride=(1, 1, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)
    best_dc = -1
    best_per_class_dcs = [-1 for i in range(9)]

    start_time = time.time()
    for epoch in range(args.num_epochs):
        epoch_time = time.time()
        trainloss = 0
        for x_train, y_train, flagvec in traindataloader:
            # goes OOM on this example
            if x_train.shape == torch.Size([1, 1, 168, 328, 232]):
                continue
            x_train = torch.autograd.Variable(x_train.cuda())
            y_train = torch.autograd.Variable(y_train.cuda())
            optimizer.zero_grad()
            pred = model(x_train)
            pred = F.softmax(pred, dim=1)
            loss = monai.losses.DiceLoss(include_background=args.include_background, squared_pred=args.squared_pred)(pred, y_train)
            loss.backward()
            optimizer.step()
            trainloss += loss.item()
            del loss, x_train, y_train, pred
        
        print(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Train Loss = {trainloss}")
        logging.info(f"Epoch [{epoch}] Time = {round(time.time()-epoch_time, 5)}")
        logging.info(f"Epoch [{epoch}] Total Time = {round(time.time()-start_time, 5)}")
        logging.info(f"Max memory allocated = {torch.cuda.max_memory_allocated()/1e9} GB")
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






