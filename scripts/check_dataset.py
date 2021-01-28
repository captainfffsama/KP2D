# -*- coding: utf-8 -*-

# @Description: 数据集检查工具
# @Author: CaptainHu
# @Date: 2021-01-28 09:33:41
# @LastEditors: CaptainHu
import argparse
import os
from concurrent import futures

from tqdm import tqdm
from PIL import Image

from kp2d.utils.config import parse_train_file
from train_keypoint_net_utils import setup_datasets_and_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="数据集检查")
    parser.add_argument("--dataset",)
    parser.add_argument("--log")
    args = parser.parse_args()

def get_all_file_path(file_dir:str,filter_=('.jpg')) -> list:
    #遍历文件夹下所有的file
    return [os.path.join(maindir,filename) for maindir,_,file_name_list in os.walk(file_dir) \
        for filename in file_name_list \
        if os.path.splitext(filename)[1] in filter_ ]

def check_img(file_path):
    try:
        img=Image.open(file_path)
    except: 
        img=None
    return img,file_path

def main(args):
    if os.path.isdir(args.dataset):
        all_path=get_all_file_path(args.dataset)
    elif os.path.isfile(args.dataset):
        with open(args.dataset,'r') as fr:
            all_path=[x.strip() for x in fr.readlines()]
    else:
        raise ValueError('dataset 不存在')
    
    error_list=[]

    with futures.ThreadPoolExecutor(64) as exec:
        tasks=[exec.submit(check_img,x) for x in all_path]
        for task in tqdm(futures.as_completed(tasks),total=len(all_path)):
            img,img_path=task.result()
            if img is None:
                error_list.append(img_path+"\n")
    
    with open(args.log,'w') as fw:
        fw.writelines(error_list)


if __name__=="__main__":
    args=parse_args()
    main(args)
