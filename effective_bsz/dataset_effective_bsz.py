# Python imports
import os, math, pdb
import numpy as np
from glob import glob
from tqdm import tqdm

# ML imports
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
import torch as t
t.backends.cudnn.enabled = True
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

# Medical imaging specific imports
import SimpleITK as sitk
import monai
from monai.transforms import RandAffined, Rand3DElasticd, RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, ToTensord, RandZoomd, Resized, Compose

from dset_utils import *

# Global variables
TRAIN_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_crp_v2_pool1.pth'
TEST_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/testpddca15_crp_v2_pool1.pth'
CET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_cet_crp_v2_pool1.pth'
PET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_pet_crp_v2_pool1.pth'
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

class DatasetStg1(Dataset):
    def __init__(self,path, istranform=True, alpha=1000, sigma=30, alpha_affine=0.04, istest=False, monai_transform=None):
        self.datas = t.load(path)
        self.ist = istranform
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.istest = istest
        self.monai_transform = monai_transform
    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy().astype(np.float32)
        if not self.istest:
            for mask in data['mask']: # for multi-task 
                if mask is None: 
                    print(data['name'])
                    assert 1 == 0
        if not self.ist: #[::2, ::2, ::2]
            masklst = []
            for mask in data['mask']:
                if mask is None: mask = np.zeros((1,img.shape[0],img.shape[1],img.shape[2])).astype(np.uint8)
                masklst.append(mask.astype(np.uint8).reshape((1,img.shape[0],img.shape[1],img.shape[2]))) 
            mask0 = np.zeros_like(masklst[0]).astype(np.uint8)
            for mask in masklst:
                mask0 = np.logical_or(mask0, mask).astype(np.uint8)
            mask0 = 1 - mask0
            img, label, flag = t.from_numpy(img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))), t.from_numpy(np.concatenate([mask0]+masklst, axis=0)), True
            if self.monai_transform=="rand_affine":
                data_dict = {"image":img, "label":label}
                transform = RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=0.5,
                #     spatial_size=(240, 240, 155),
                    translate_range=(5, 15, 15),
                    rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                    scale_range=(0.1, 0.1, 0.1),
                    padding_mode="border",
                #     device=torch.device("cuda:0")
                )
                transformed_data_dict = transform(data_dict)
                return transformed_data_dict['image'], transformed_data_dict['label'], flag
            elif self.monai_transform=="rand_elastic":
                data_dict = {"image":img, "label":label}
                transform = Rand3DElasticd(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=0.5,
                    sigma_range=(5, 8),
                    magnitude_range=(10, 20),
                #     spatial_size=(50, 50, 50),
                    translate_range=(5, 15, 15),
                    rotate_range=(np.pi / 18, np.pi / 18, np.pi/18),
                    scale_range=(0.1, 0.1, 0.1),
                    padding_mode="border",
                )
                transformed_data_dict = transform(data_dict)
                return transformed_data_dict['image'], transformed_data_dict['label'], flag
            elif self.monai_transform=="rand_spatial_crop":
                data_dict = {"image":img, "label":label}
                crop_factor = 0.9
                size = [int(crop_factor*data_dict['image'].shape[1]), int(crop_factor*data_dict['image'].shape[2]), int(crop_factor*data_dict['image'].shape[3])]
                transform = Compose([
                    RandSpatialCropd(keys=["image", "label"], roi_size=size, random_size=False),
                    Resized(keys=["image", "label"], spatial_size=(data_dict['image'].shape[1], data_dict['image'].shape[2], data_dict['image'].shape[3])),
                ]
                )
                transformed_data_dict = transform(data_dict)
                return transformed_data_dict['image'], transformed_data_dict['label'], flag
            elif self.monai_transform=="crop_same_size":
                data_dict = {"image":img, "label":label}
                size = [78, 206, 164] # average along all 3 dims in traindataset
                transform = Compose([
                    Resized(keys=["image", "label"], spatial_size=size),
                ]
                )
                transformed_data_dict = transform(data_dict)
                return transformed_data_dict['image'], transformed_data_dict['label'], flag
            elif self.monai_transform=="rand_zoom":
                data_dict = {"image":img, "label":label}
                transform = RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2)
                transformed_data_dict = transform(data_dict)
                return transformed_data_dict['image'], transformed_data_dict['label'], flag           
            else:
                return img, label, flag

        im_merge = np.concatenate([img[...,None]]+[mask.astype(np.float32)[...,None] for mask in data['mask']],                                  axis=3)
        # Apply transformation on image
        # im_merge_t, new_img = self.elastic_transform3Dv2(im_merge,self.alpha,self.sigma,min(im_merge.shape[1:-1])*self.alpha_affine)
        im_merge_t = im_merge
        # Split image and mask ::2, ::2, ::2
        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[..., 1:].astype(np.uint8).transpose(3, 0, 1, 2)
        mask0 = np.zeros_like(im_mask_t[0, :, :, :]).reshape((1,)+im_mask_t.shape[1:]).astype(np.uint8)
        im_mask_t_lst = []
        flagvect = np.ones((10,), np.float32)
        retflag = True
        for i in range(9):
            im_mask_t_lst.append(im_mask_t[i,:,:,:].reshape((1,)+im_mask_t.shape[1:]))
            if im_mask_t[i,:,:,:].max() != 1: 
                retflag = False
                flagvect[i+1] = 0
            mask0 = np.logical_or(mask0, im_mask_t[i,:,:,:]).astype(np.uint8)
        if not retflag: flagvect[0] = 0
        mask0 = 1 - mask0
        img, label, flag = t.from_numpy(im_t.reshape((1,)+im_t.shape[:3])), t.from_numpy(np.concatenate([mask0]+im_mask_t_lst, axis=0)), flagvect
        if self.monai_transform=="rand_affine":
            data_dict = {"image":img, "label":label}
            transform = RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
            #     spatial_size=(240, 240, 155),
                translate_range=(5, 15, 15),
                rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            #     device=torch.device("cuda:0")
            )
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag
        elif self.monai_transform=="rand_elastic":
            data_dict = {"image":img, "label":label}
            transform = Rand3DElasticd(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.5,
                sigma_range=(5, 8),
                magnitude_range=(10, 20),
            #     spatial_size=(50, 50, 50),
                translate_range=(5, 15, 15),
                rotate_range=(np.pi / 18, np.pi / 18, np.pi/18),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            )
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag
        elif self.monai_transform=="rand_spatial_crop":
            data_dict = {"image":img, "label":label}
            crop_factor = 0.9
            size = [int(crop_factor*data_dict['image'].shape[1]), int(crop_factor*data_dict['image'].shape[2]), int(crop_factor*data_dict['image'].shape[3])]
            transform = Compose([
                RandSpatialCropd(keys=["image", "label"], roi_size=size, random_size=False),
                Resized(keys=["image", "label"], spatial_size=(data_dict['image'].shape[1], data_dict['image'].shape[2], data_dict['image'].shape[3])),
            ]
            )
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag
        elif self.monai_transform=="crop_same_size":
            data_dict = {"image":img, "label":label}
            size = [78, 206, 164] # average along all 3 dims in traindataset
            transform = Compose([
                Resized(keys=["image", "label"], spatial_size=size),
            ]
            )
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag
        elif self.monai_transform=="rand_zoom":
            data_dict = {"image":img, "label":label}
            transform = RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.8, max_zoom=1.2)
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag           
        else:
            return img, label, flag
    def __len__(self):
        return len(self.datas)
    def elastic_transform3Dv2(self, image, alpha, sigma, alpha_affine, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.
         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
         From https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        # affine and deformation must be slice by slice and fixed for slices
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = image.shape # image is contatenated, the first channel [:,:,:,0] is the image, the second channel 
        # [:,:,:,1] is the mask. The two channel are under the same tranformation.
        shape_size = shape[:-1] # z y x
        # Random affine
        shape_size_aff = shape[1:-1] # y x
        center_square = np.float32(shape_size_aff) // 2
        square_size = min(shape_size_aff) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size],                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        new_img = np.zeros_like(image)
        for i in range(shape[0]):
            new_img[i,:,:,0] = cv2.warpAffine(image[i,:,:,0], M, shape_size_aff[::-1],                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
            for j in range(1, 10):
                new_img[i,:,:,j] = cv2.warpAffine(image[i,:,:,j], M, shape_size_aff[::-1], flags=cv2.INTER_NEAREST,                                                  borderMode=cv2.BORDER_TRANSPARENT, borderValue=0)
        dx = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape[1:-1]) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape_size_aff[1]), np.arange(shape_size_aff[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        new_img2 = np.zeros_like(image)
        for i in range(shape[0]):
            new_img2[i,:,:,0] = map_coordinates(new_img[i,:,:,0], indices, order=1, mode='constant').reshape(shape[1:-1])
            for j in range(1, 10):
                new_img2[i,:,:,j] = map_coordinates(new_img[i,:,:,j], indices, order=0, mode='constant').reshape(shape[1:-1])
        return np.array(new_img2), new_img

if __name__ == "__main__":
    transform = monai.transforms.RandAffined(
        keys=["image", "label"],
        mode=("bilinear", "nearest"),
        prob=1.0,
    #     spatial_size=(240, 240, 155),
        translate_range=(2, 10, 10),
        rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        scale_range=(0.1, 0.1, 0.1),
        padding_mode="border",
    #     device=torch.device("cuda:0")
    )
    traindataset = DatasetStg1(PET_PATH, istranform=True, monai_transform=transform)
    traindataloader = t.utils.data.DataLoader(traindataset,num_workers=0,batch_size=1, shuffle=True)
    testdataset = DatasetStg1(TEST_PATH, istranform=False)
    testdataloader = t.utils.data.DataLoader(testdataset,num_workers=0,batch_size=1)
    print(len(traindataloader), len(testdataloader))
    traindataset[0]
    pdb.set_trace()