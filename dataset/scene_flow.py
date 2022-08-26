#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import os

from copy import deepcopy
import numpy as np
import torch.utils.data as data
from torchvision.transforms import ToTensor
from PIL import Image
from albumentations import Compose, OneOf
from natsort import natsorted

from dataset.preprocess import augment, augment_disparity_change
from dataset.stereo_albumentation import random_crop, horizontal_flip, UnifiedMyCenterCropWithDispChange, \
    random_location, crop, RandomShiftRotate, RGBShiftStereo, GaussNoiseStereo, RandomBrightnessContrastStereo, horizontal_flip_disparity_change
from utilities.python_pfm import readPFM
from utilities.integration_tools import Arguments


class SceneFlowSamplePackDataset(data.Dataset):
    def __init__(self, datadir, args: Arguments, split='train'):
        super(SceneFlowSamplePackDataset, self).__init__()

        self.datadir = datadir
        self.left_fold = 'RGBcleanpass/left/'
        self.right_fold = 'RGBcleanpass/right/'

        self.disp = 'disparity/left'
        self.occ_fold = 'occlusion/left'
        self.disp_change = 'disp_change/left'

        self.disp_right = 'disparity/right'
        self.occ_fold_right = 'occlusion/right'
        self.disp_change_right = 'disp_change/right'
        self.split = split

        self.args = args
        self.totensor = ToTensor()
        self.split = split
        if self.args.dataset_max_length is not None:
            # global index
            self.data = os.listdir(os.path.join(self.datadir, self.left_fold))[:self.args.dataset_max_length]
        else:
            self.data = os.listdir(os.path.join(self.datadir, self.left_fold))

        if split == 'validation':
            self.data = self.data[:int(len(self.data) * self.args.eval_percentage)]
        else:
            self.data = self.data[:int(len(self.data) * (1-self.args.eval_percentage))]
        self._augmentation()

    def _augmentation(self):
        if self.split == 'train':
            self.transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.3),
            #     OneOf([
            #         GaussNoiseStereo(always_apply=True, p_asym=1.0),
            #         RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
            #     ], p=1.0)
             ])
        else:
            self.transformation = None
        
        self.transformation = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = {}
        path = self.datadir
        # try to read data from disp_change, using tray-except to handle the discontinuity
        # for example, in the 0 to 9 frame, when read the 9th frame's index, we can't find it's disp_change file
        # so we try to catch this error and return a dict with key "Error"
        try:
            if self.split == 'train':
                # manully random crop the data
                x1, y1, x2, y2 = random_location(self.args.height, self.args.width,
                                                 (self.args.origin_height, self.args.origin_width), self.split)
            else:
                y1 = 142
                x1 = 252
                y2 = 398
                x2 = 707

            # we only compute forward disparity change, if you need backward, please add you own floder
            disp_change, _ = readPFM(os.path.join(path, self.disp_change, self.data[idx].replace('png', 'pfm')))
            if x1 is not None:
                input_data['disp_change'] = crop(disp_change, x1, y1, x2, y2).copy()
            else:
                input_data['disp_change'] = disp_change.copy()
            
            disp_change_right, _ = readPFM(os.path.join(path, self.disp_change_right, self.data[idx].replace('png', 'pfm')))
            if x1 is not None:
                input_data['disp_change_right'] = crop(disp_change_right, x1, y1, x2, y2).copy()
            else:
                input_data['disp_change_right'] = disp_change_right.copy()
        
            left_t1 = np.array(Image.open(os.path.join(path, self.left_fold, self.data[idx]))).astype(np.uint8)[..., :3]
            if x1 is not None:
                input_data['left_t1'] = crop(left_t1, x1, y1, x2, y2).copy()
            else:
                input_data['left_t1'] = left_t1.copy()

            left_t2 = np.array(Image.open(os.path.join(path, self.left_fold, self.data[idx + 1]))).astype(np.uint8)[...,
                      :3]
            if x1 is not None:
                input_data['left_t2'] = crop(left_t2, x1, y1, x2, y2).copy()
            else:
                input_data['left_t2'] = left_t2.copy()

            right_t1 = np.array(Image.open(os.path.join(path, self.right_fold, self.data[idx]))).astype(np.uint8)[...,
                       :3]
            if x1 is not None:
                input_data['right_t1'] = crop(right_t1, x1, y1, x2, y2).copy()
            else:
                input_data['right_t1'] = right_t1.copy()

            right_t2 = np.array(Image.open(os.path.join(path, self.right_fold, self.data[idx + 1]))).astype(np.uint8)[
                       ..., :3]
            if x1 is not None:
                input_data['right_t2'] = crop(right_t2, x1, y1, x2, y2).copy()
            else:
                input_data['right_t2'] = right_t2.copy()

            occ_t1 = np.array(Image.open(os.path.join(path, self.occ_fold, self.data[idx]))).astype(np.bool)
            if x1 is not None:
                input_data['occ_mask_t1'] = crop(occ_t1, x1, y1, x2, y2).copy()
            else:
                input_data['occ_mask_t1'] = occ_t1.copy()

            occ_t2 = np.array(Image.open(os.path.join(path, self.occ_fold, self.data[idx + 1]))).astype(np.bool)
            if x1 is not None:
                input_data['occ_mask_t2'] = crop(occ_t2, x1, y1, x2, y2).copy()
            else:
                input_data['occ_mask_t2'] = occ_t2.copy()
            
            occ_t1_right = np.array(Image.open(os.path.join(path, self.occ_fold_right, self.data[idx]))).astype(np.bool)
            if x1 is not None:
                input_data['occ_mask_right_t1'] = crop(occ_t1_right, x1, y1, x2, y2).copy()
            else:
                input_data['occ_mask_right_t1'] = occ_t1_right.copy()

            occ_t2_right = np.array(Image.open(os.path.join(path, self.occ_fold_right, self.data[idx + 1]))).astype(np.bool)
            if x1 is not None:
                input_data['occ_mask_right_t2'] = crop(occ_t2_right, x1, y1, x2, y2).copy()
            else:
                input_data['occ_mask_right_t2'] = occ_t2_right.copy()

            disp_t1, _ = readPFM(os.path.join(path, self.disp, self.data[idx].replace('png', 'pfm')))
            if x1 is not None:
                input_data['disp_t1'] = crop(disp_t1, x1, y1, x2, y2).copy()
            else:
                input_data['disp_t1'] = disp_t1.copy()

            disp_t2, _ = readPFM(os.path.join(path, self.disp, self.data[idx + 1].replace('png', 'pfm')))   
            if x1 is not None:
                input_data['disp_t2'] = crop(disp_t2, x1, y1, x2, y2).copy()
            else:
                input_data['disp_t2'] = disp_t2.copy()
            
            disp_t1_right, _ = readPFM(os.path.join(path, self.disp_right, self.data[idx].replace('png', 'pfm')))
            if x1 is not None:
                input_data['disp_t1_right'] = crop(disp_t1_right, x1, y1, x2, y2).copy()
            else:
                input_data['disp_t1_right'] = disp_t1_right.copy()

            disp_t2_right, _ = readPFM(os.path.join(path, self.disp_right, self.data[idx + 1].replace('png', 'pfm')))   
            if x1 is not None:
                input_data['disp_t2_right'] = crop(disp_t2_right, x1, y1, x2, y2).copy()
            else:
                input_data['disp_t2_right'] = disp_t2_right.copy()
            

            # horizontal flip
            # img_left, img_right, occ_left, occ_right, disp_left, disp_right, split
            input_data['left_t1'], input_data['right_t1'], input_data['occ_mask_t1'], input_data['occ_mask_right_t1'], input_data['disp_t1'], input_data['disp_t1_right'] = horizontal_flip(input_data['left_t1'], input_data['right_t1'], input_data['occ_mask_t1'], 
            input_data['occ_mask_right_t1'], input_data['disp_t1'], input_data['disp_t1_right'],self.split)

            input_data['left_t2'], input_data['right_t2'], input_data['occ_mask_t2'], input_data['occ_mask_right_t2'], input_data['disp_t2'], input_data['disp_t2_right'] = horizontal_flip(input_data['left_t2'], input_data['right_t2'], input_data['occ_mask_t2'], 
            input_data['occ_mask_right_t2'], input_data['disp_t2'], input_data['disp_t2_right'],self.split)

            input_data['disp_change'], input_data['disp_change_right'] = horizontal_flip_disparity_change(np.ascontiguousarray(input_data['disp_change']), 
                                                                                                            np.ascontiguousarray(input_data['disp_change_right']), self.split)

            input_data = augment_disparity_change(input_data, self.transformation, self.args)

            return input_data

        except:
            left_t1 = np.array(Image.open(os.path.join(path, self.left_fold, self.data[idx]))).astype(np.uint8)[..., :3]
            # we must reshape this ndarray, to adjust the channel
            input_data['Error'] = self.totensor(np.ascontiguousarray(left_t1.reshape(-1, 540, 960)))
            return input_data

