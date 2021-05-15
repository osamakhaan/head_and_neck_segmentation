from glob import glob
import SimpleITK as sitk
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os
import numpy as np
from tqdm import tqdm

import torch as t
t.backends.cudnn.enabled = True
from torch.utils import data
import pdb
import monai
from monai.transforms import RandAffined, Rand3DElasticd, RandSpatialCropd, RandFlipd, NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd, ToTensord, RandZoomd, Resized, Compose

TRAIN_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_crp_v2_pool1.pth'
TEST_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/testpddca15_crp_v2_pool1.pth'
CET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_cet_crp_v2_pool1.pth'
PET_PATH = '/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/data/trainpddca15_pet_crp_v2_pool1.pth'

import SimpleITK as sitk
import math
from scipy.ndimage.interpolation import zoom

from torch.utils.data import Dataset, DataLoader
def getdatamaskfilenames(path, maskname):
    data, masks_data = [], []
    for pth in path: # get data files and mask files
        maskfiles = []
        for seg in maskname:
            if os.path.exists(os.path.join(pth, './structures/'+seg+'_crp_v2.npy')):
                maskfiles.append(os.path.join(pth, './structures/'+seg+'_crp_v2.npy'))
            else:
                print('missing annotation', seg, pth.split('/')[-1])
                maskfiles.append(None)
        data.append(os.path.join(pth, 'img_crp_v2.npy'))
        masks_data.append(maskfiles)
    return data, masks_data
def imfit(img, newz, newy, newx):
    z, y, x = img.shape
    retimg = np.zeros((newz, newy, newx), img.dtype)
    bz, ez = int(newz/2), int(newz/2+1)
    while ez - bz < z:
        if bz - 1 >= 0:
            bz -= 1
        if ez - bz < z:
            if ez + 1 <= z:
                ez += 1
    by, ey = int(newy/2), int(newy/2+1)
    while ey - by < y:
        if by - 1 >= 0:
            by -= 1
        if ey - by < y:
            if ey + 1 <= y:
                ey += 1
    bx, ex = int(newx/2), int(newx/2+1)
    while ex - bx < x:
        if bx - 1 >= 0:
            bx -= 1
        if ex - bx < x:
            if ex + 1 <= x:
                ex += 1
    # pdb.set_trace()
    retimg[bz:ez, by:ey, bx:ex] = img
    return retimg
def getdatamask(data, mask_data, debug=False): # read data and mask, reshape
    datas = []
    for fnm, masks in tqdm(zip(data, mask_data)):
        item = {}
        img = np.load(fnm) # z y x
        nz, ny, nx = img.shape
        tnz, tny, tnx = math.ceil(nz/8.)*8., math.ceil(ny/8.)*8., math.ceil(nx/8.)*8.
        img = imfit(img, int(tnz), int(tny), int(tnx)) #zoom(img, (tnz/nz,tny/ny,tnx/nx), order=2, mode='nearest')
        item['img'] = t.from_numpy(img)
        item['mask'] = []
        for idx, maskfnm in enumerate(masks):
            if maskfnm is None: 
                ms = np.zeros((nz, ny, nx), np.uint8)
            else: 
                ms = np.load(maskfnm).astype(np.uint8)
                assert ms.min() == 0 and ms.max() == 1
            mask = imfit(ms, int(tnz), int(tny), int(tnx)) #zoom(ms, (tnz/nz,tny/ny,tnx/nx), order=0, mode='constant')
            item['mask'].append(mask)
        assert len(item['mask']) == 9
        item['name'] = str(fnm)#.split('/')[-1]
        datas.append(item)
    return datas
def process(path='/data/wtzhu/dataset/pddca18/', debug=False):
    trfnmlst, trfnmlstopt, tefnmlstoff, tefnmlst = [], [], [], [] # get train and test files
    train_files, train_filesopt, test_filesoff, test_files = [], [], [], [] # MICCAI15 and MICCAI16 use different test
    for pid in os.listdir(path):
        if '0522c0001' <= pid <= '0522c0328':
            trfnmlst.append(pid)
            train_files.append(os.path.join(path, pid))
        elif '0522c0329' <= pid <= '0522c0479':
            trfnmlstopt.append(pid)
            train_filesopt.append(os.path.join(path, pid))
        elif '0522c0555' <= pid <= '0522c0746':
            tefnmlstoff.append(pid)
            test_filesoff.append(os.path.join(path, pid))
        elif '0522c0788' <= pid <= '0522c0878':
            tefnmlst.append(pid)
            test_files.append(os.path.join(path, pid))
        else:
            print(pid)
            assert 1 == 0
    print('train file names', trfnmlst)
    print('optional train file names', trfnmlstopt)
    print('offsite test file names', tefnmlstoff)
    print('onsite test file names', tefnmlst)
    print('Total train files', len(train_files), 'total test files', len(test_files))
    print('Train optional files', len(train_filesopt), 'test optional files', len(test_filesoff))
    assert len(trfnmlst) == 25 and len(trfnmlstopt) == 8 and len(tefnmlstoff) == 10 and len(tefnmlst) == 5
    assert len(train_files) == 25 and len(train_filesopt) == 8 and len(test_filesoff) == 10 and     len(test_files) == 5
    structurefnmlst = ('BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R',                        'Submandibular_L', 'Submandibular_R')
    train_data, train_masks_data = getdatamaskfilenames(train_files, structurefnmlst)
    train_dataopt, train_masks_dataopt = getdatamaskfilenames(train_filesopt, structurefnmlst)
    test_data, test_masks_data = getdatamaskfilenames(test_files, structurefnmlst)
    test_dataoff, test_masks_dataoff = getdatamaskfilenames(test_filesoff, structurefnmlst)
    return getdatamask(train_data+train_dataopt+test_data,                        train_masks_data+train_masks_dataopt+test_masks_data,debug=debug),            getdatamask(test_dataoff, test_masks_dataoff,debug=debug)
def processCET(path='/data/wtzhu/dataset/HNCetuximabclean/', debug=False):
    trfnmlst = [] # get train and test files
    train_files = [] # MICCAI15 and MICCAI16 use different test
    for pid in os.listdir(path):
        trfnmlst.append(pid)
        train_files.append(os.path.join(path, pid))
    print('train file names', trfnmlst)
    print('Total train files', len(train_files))
    structurefnmlst = ('BrainStem', 'Chiasm', 'Mandible', 'OpticNerve_L', 'OpticNerve_R', 'Parotid_L', 'Parotid_R',                        'Submandibular_L', 'Submandibular_R')
    train_data, train_masks_data = getdatamaskfilenames(train_files, structurefnmlst)
    return getdatamask(train_data, train_masks_data,debug=debug)
# You can skip this if you have alreadly done it.
if not os.path.isfile(TRAIN_PATH):
    train_data, test_data = process('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    t.save(train_data, TRAIN_PATH)
    t.save(test_data, TEST_PATH)
if not os.path.isfile(CET_PATH):
    train_data, test_data = process('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    data = processCET('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/HNCetuximabclean/')
    print('use ', len(data))
    t.save(data+train_data, CET_PATH)
if not os.path.isfile(PET_PATH):
    train_data, test_data = process('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/pddca18/')
    print('use train', len(train_data), 'use test', len(test_data))
    data = processCET('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/HNCetuximabclean/')
    print('use ', len(data))
    petdata = processCET('/scratch/mok232/Medical-Imaging/AnatomyNet-for-anatomical-segmentation/HNPETCTclean/')
    print('use ', len(petdata))
    t.save(data+train_data+petdata, PET_PATH)

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
            data_dict = {"image":img, "label":label}
            if self.monai_transform == "crop1":
                size = [64, 128, 128]
            elif self.monai_transform == "crop2":
                size = [128, 128, 64]
            transform = Compose([
                Resized(keys=["image", "label"], spatial_size=size),
            ]
            )
            transformed_data_dict = transform(data_dict)
            return transformed_data_dict['image'], transformed_data_dict['label'], flag

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
        data_dict = {"image":img, "label":label}
        if self.monai_transform == "crop1":
            size = [64, 128, 128]
        elif self.monai_transform == "crop2":
            size = [128, 128, 64]
        transform = Compose([
            Resized(keys=["image", "label"], spatial_size=size),
        ]
        )
        transformed_data_dict = transform(data_dict)
        return transformed_data_dict['image'], transformed_data_dict['label'], flag
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