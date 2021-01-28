# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 01月 25日 星期一 10:57:53 CST
@Description: 用来debug的工具
'''

import numpy as np
import matplotlib.pyplot as plt
import torch

def show_img(img,kps=None):
    if isinstance(img,torch.Tensor):
        img=img.cpu().numpy()
    if kps:
        img=draw_kp(kps,img)
    plt.imshow(img)
    plt.show()

def draw_kp(kps,img):
    """
        Args:
            kps: np.ndarray
                Nx2 排列为xy
    """
    for i in kps:
        img=cv2.circle(img,i[:2],2,(255,0,0),3)
    return img

