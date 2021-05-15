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


# Global variables
TRAIN_PATH = '/scratch/mma525/dl-system/dl_systems_project/data/trainpddca15_crp_v2_pool1.pth'
TEST_PATH = '/scratch/mma525/dl-system/dl_systems_project/data/testpddca15_crp_v2_pool1.pth'
CET_PATH = '/scratch/mma525/dl-system/dl_systems_project/data/trainpddca15_cet_crp_v2_pool1.pth'
PET_PATH = '/scratch/mma525/dl-system/dl_systems_project/data/trainpddca15_pet_crp_v2_pool1.pth'



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