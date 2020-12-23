# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2020-12-22 14:13:34
# @LastEditors: CaptainHu
from ipdb import set_trace

import cv2
import numpy as np

from kp2d.datasets.patches_dataset import PatchesDataset
from infer_demo import Homographier,KeyPointModel

def get_orb(im1_path,im2_path):
    im1=cv2.imread(im1_path,cv2.IMREAD_COLOR)
    im2=cv2.imread(im2_path,cv2.IMREAD_COLOR)

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(1000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    kp1_array= np.array([list(kp.pt) for kp in keypoints1])
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    kp2_array= np.array([list(kp.pt) for kp in keypoints2])
    return kp1_array, descriptors1,kp2_array,descriptors2



if __name__=='__main__':
    image_list=[
        r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/35kV消弧线圈本体/20200908_10_20_08_1599531621344/35kV消弧线圈油位表.jpg',
        r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/35kV消弧线圈本体/20201102_15_00_02_1604300402835/35kV消弧线圈油位表.jpg',
    ]

    kp1,d1,kp2,d2=get_orb(*image_list)
    matcher=Homographier()
    H=matcher(kp1,d1,kp2,d2)
    print(H)