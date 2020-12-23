# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2020年 12月 23日 星期三 15:25:13 CST
@Description: 用来生成对位用的测试数据集
'''
import os
import sys

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import random

import cv2
import albumentations as A
import numpy as np

import test_dataset_generate.homo_config as cfg

from test_tools import pdb_show_img

from ipdb import set_trace

def get_transform():
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
    ])
    return transform

def get_all_jpg_path(file_dir:str,filter_:tuple=('.jpg',)) -> list:
    #遍历文件夹下所有的jpg
    if os.path.isdir(file_dir):
        return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
                for filename in file_name_list \
                if os.path.splitext(filename)[1] in filter_ ]
    elif os.path.isfile(file_dir):
        with open(file_dir,'r') as fr:
            file_list=[x.strip() for x in fr.readlines if os.path.splitext(x.strip())[-1] in filter_]
        return file_list
    else:
        raise FileNotFoundError('{} have no image in {}'.format(file_dir,filter_))

def generat_one_group(ori_img_path:str,transform):
    ori_img=cv2.imread(ori_img_path,cv2.IMREAD_COLOR)
    ori_img_rgb=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    h,w=ori_img.shape[:-1]
    if max(w/7,h/7) <= cfg.homo_range:
        print('{} is too small,please change cfg.homo_range'.format(ori_img_path))
        return

    ori_kp=(int(w/7),int(h/7),
            int(w*6/7),int(h/7),
            int(w/7),int(h*6/7),
            int(w*6/7),int(h*6/7),)

    ori_kp=np.array(ori_kp,dtype="float32").reshape(4,2)

    file_name=os.path.basename(ori_img_path).split('.')[0]
    save_dir=os.path.join(cfg.dataset_save,file_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_img=ori_img[int(h/7):int(h*6/7),int(w/7):int(w*6/7)]
    cv2.imwrite(os.path.join(save_dir,'1.jpg'),img=ori_img[int(h/7):int(h*6/7),int(w/7):int(w*6/7)])

    for idx in range(2,cfg.img_num_per_group+2):
        warp_img=transform(image=ori_img_rgb)['image']
        warp_img=cv2.cvtColor(warp_img, cv2.COLOR_RGB2BGR)
        
        # 生成homography
        warp_kp=(ori_kp+np.random.randint(-cfg.homo_range,cfg.homo_range,size=ori_kp.shape)).astype(np.float32)

        H=cv2.getPerspectiveTransform(ori_kp,warp_kp)
        final_warp_img=cv2.warpPerspective(warp_img,H,(w,h))

        #保存结果
        np.savetxt(os.path.join(save_dir,'H_1_{}'.format(idx+1)),H,delimiter=' ')
        cv2.imwrite(os.path.join(save_dir,'{}.jpg'.format(idx+1)),img=final_warp_img[int(h/7):int(h*6/7),int(w/7):int(w*6/7)])
        

def main():
    jpg_list=get_all_jpg_path(cfg.imgs_path)
    if not os.path.exists(cfg.dataset_save):
        os.mkdir(cfg.dataset_save)
    print('start select image')

    if len(jpg_list)>cfg.generate_group_num:
        jpg_list=random.sample(jpg_list,k=cfg.generate_group_num)
    else:
        jpg_list=random.choices(jpg_list,k=cfg.generate_group_num)

    transform=get_transform()
    
    for jpg in jpg_list:
        generat_one_group(jpg,transform)

if __name__=="__main__":
    main()
    
