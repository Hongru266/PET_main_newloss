import glob
import os
import cv2
import json
import numpy as np
from PIL import Image
import scipy.io as io
from scipy.io import savemat


image_old_list = "data/UCF-QNRF/Train/*.jpg"
gt_old_list = "data/UCF-QNRF/Train/*.mat"
dict_old_list = "data/UCF-QNRF/dict/*.json"
mask_old_list = "data/UCF-QNRF/masks/*.npy"

image_list = "data/UCF-QNRF_1536/Train/*.jpg"
gt_list = "data/UCF-QNRF_1536/Train/*.mat"
dict_list = "data/UCF-QNRF_1536/dict/*.json"
mask_list = "data/UCF-QNRF_1536/masks/*.npy"
count=0

N = len(image_list)
for i in range(N):
    if count > 10:
        break
    image_old = cv2.imread(sorted(glob.glob(image_old_list))[i])
    image = cv2.imread(sorted(glob.glob(image_list))[i])
    gt_old = io.loadmat(sorted(glob.glob(gt_old_list))[i])
    gt = io.loadmat(sorted(glob.glob(gt_old_list))[i])
    dicts = json.load(open(sorted(glob.glob(dict_list))[i], 'r', encoding='utf-8'))
    masks_old = np.load(sorted(glob.glob(mask_old_list))[i])
    masks = np.load(sorted(glob.glob(mask_list))[i])
    print(f"image_old: {image_old.shape}, image shape: {image.shape}, gt_old points shape:{gt_old['annPoints'].shape} gt points num: {gt['annPoints'].shape}, dicts num: {len(dicts)}, masks_old shape:{masks_old.shape}, masks shape: {masks.shape}")
    count += 1