# -*- coding: utf-8 -*-

# @Description: 用来启动评测器的
# @Author: CaptainHu
# @Date: 2020-12-24 16:40:27
# @LastEditors: CaptainHu
import argparse
from collections import defaultdict

import context

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from infer_demo import KeyPointModel,Homographier
from kp2d.utils.image import to_color_normalized
from kp2d.evaluation.descriptor_evaluation import select_k_best
from dataset import EvalDataset

from ipdb import set_trace

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='keypoint eval')
    parser.add_argument('--file', default=r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/models/kp2d/v4.ckpt',
    type=str, help='Input file (.ckpt or .yaml)')
    parser.add_argument('--dataset',default=r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/kp2d/HPatches',
    type=str)
    parser.add_argument('--out_size',default=(640,480),type=tuple,)
    parser.add_argument('--k',default=5,type=int,help='用来控制各项指标的阈值,一般是1,3,5')
    args = parser.parse_args()
    return args

class CVKPDetector(object):
    def __init__(self,detector_type:str,matcher_param:dict=None):
        detec_map={
            'orb':cv2.ORB_create,
            'sift':cv2.SIFT_create,
            'akaze':cv2.AKAZE_create,
        }
        if matcher_param:
            self.detector=detec_map[detector_type](**matcher_param)
        else:
            self.detector=detec_map[detector_type]()
    
    def get_kp(self,img):
        if img.shape[-1] ==3:
            img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps,desc=self.detector.detectAndCompute(img,None)
        kps=np.array([list(kp.pt) for kp in kps])
        return kps,desc

    def __call__(self,src,warp_src):
        kp,desc=self.get_kp(src)
        w_kp,w_desc=self.get_kp(warp_src)
        return kp,desc,w_kp,w_desc

class KPNet(KeyPointModel):
    def __call__(self,image_t,warped_image_t):
        with torch.no_grad():
            image=to_color_normalized(image_t).cuda()
            warped_image=to_color_normalized(warped_image_t).cuda()

            '''
            score shape is (N,1,H/8,W/8)
            coord shape is (N,2,H/8,W/8)
            desc shape is (N,256,H/8,W/8)
            '''
            score_1, coord_1, desc1 = self.keypoint_net(image)
            score_2, coord_2, desc2 = self.keypoint_net(warped_image)
            B, _, Hc, Wc = desc1.shape

            # Scores & Descriptors
            # prob is (x,y,score)
            prob1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            prob2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            desc2 = desc2.view(256, Hc, Wc).view(256, -1).t().cpu().numpy()
            
            # Filter based on confidence threshold
            desc1 = desc1[prob1[:, 2] > self.conf_threshold, :]
            desc2 = desc2[prob2[:, 2] > self.conf_threshold, :]
            prob1 = prob1[prob1[:, 2] > self.conf_threshold, :]
            prob2 = prob2[prob2[:, 2] > self.conf_threshold, :]
            
            desc = desc1
            warped_desc = desc2
            keypoints,desc = select_k_best(prob1, desc, self.eval_params['top_k'])
            warped_keypoints, warped_desc = select_k_best(prob2, warped_desc, self.eval_params['top_k'])
        return keypoints, desc, warped_keypoints, warped_desc

def transform_kp(points,H):
    r'''对kp进行单应性变化

        Args:
            points: np.ndarray
                Kx2 的矩阵,会被拓展乘 Kx3 大小
            H: np.ndarray
                3x3 的单应性矩阵
        Returns:
            np.array: Kx2 大小的矩阵,变换之后的点集
    '''
    points=np.insert(points,2,1,axis=1)
    new_points=points@H.T
    return new_points[:,:2]/new_points[:,2:]

def filter_keypoints(points, shape) -> tuple:
    """ Keep only the points whose coordinates are inside the dimensions of shape. """
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
            (points[:, 1] >= 0) & (points[:, 1] < shape[1])
    return points[mask, :],mask

def compute_repeatability(kp1,kp2,gt_H,img_shape,distance_thresh=3):
    # 将kp1 变换到 kp2 的座标系
    kp2_gt=transform_kp(kp1[:,:2],gt_H)
    kp2_gt,_=filter_keypoints(kp2_gt,img_shape)

    # 计算kp2和kp2_gt的一对一关系
    N1=kp2.shape[0]
    N2=kp2_gt.shape[0]
    kp2_gt=np.expand_dims(kp2_gt,1)
    kp2=np.expand_dims(kp2,0)
    norm=np.linalg.norm(kp2_gt-kp2,axis=2)

    count1 = 0
    count2 = 0
    le1 = 0
    le2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        correct1 = (min1 <= distance_thresh)
        count1 = np.sum(correct1)
        le1 = min1[correct1].sum()
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        correct2 = (min2 <= distance_thresh)
        count2 = np.sum(correct2)
        le2 = min2[correct2].sum()
    if N1 + N2 > 0 and count1 + count2 >0:
        repeatability = (count1 + count2) / (N1 + N2)
        loc_err = (le1 + le2) / (count1 + count2)
    else:
        repeatability = -1
        loc_err = -1

    return N1, N2, repeatability, loc_err

def compute_cor_k(image_shape,gt_H,pre_H,k):
    corners = np.array([[0, 0],
                        [0, image_shape[1] - 1],
                        [image_shape[0] - 1, 0],
                        [image_shape[0] - 1, image_shape[1] - 1]])
    gt_warp_corner=transform_kp(corners,gt_H)
    warp_corner=transform_kp(corners,pre_H)
    mean_dist = np.mean(np.linalg.norm(gt_warp_corner - warp_corner, axis=1))
    correctness=float(mean_dist<=k)
    return correctness

def compute_M_score(kp1,disc1,kp2,disc2,gt_H,out_shape,dis_thr=3):
    r'''将kp2映射到kp1座标系上,然后保留仅在图片上是点,计算两者距离差异.然后将小于阈值的点数目比上总点数
        作为分数,然后反过来将kp1映射到kp2再计算一遍,两个分数平均作为最终分数.
    '''
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matcher=bf.match(disc1,disc2)
    matches_idx = np.array([m.queryIdx for m in matcher])
    m_keypoints = kp1[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matcher])
    m_warped_keypoints = kp2[matches_idx, :]

    gt_warped_kp=transform_kp(m_warped_keypoints,np.linalg.inv(gt_H))
    vis_warped = np.all((gt_warped_kp >= 0) & (gt_warped_kp <= (np.array(out_shape)-1)), axis=-1)
    norm1 = np.linalg.norm(gt_warped_kp - m_keypoints, axis=-1)
    correct1 = (norm1 < dis_thr)
    count1 = np.sum(correct1 * vis_warped)
    score1 = count1 / np.maximum(np.sum(vis_warped), 1.0)

    matcher=bf.match(disc2,disc1)
    matches_idx = np.array([m.queryIdx for m in matcher])
    m_warped_keypoints = kp2[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in matcher])
    m_keypoints = kp1[matches_idx, :]

    gt_kp=transform_kp(m_keypoints,gt_H)
    vis = np.all((gt_kp >= 0) & (gt_kp <= (np.array(out_shape)-1)), axis=-1)
    norm2=np.linalg.norm(gt_kp-m_warped_keypoints,axis=-1)
    correct2 = (norm2<dis_thr)
    count2=np.sum(correct2*vis)
    score2 = count2 / np.maximum(np.sum(vis), 1.0)

    ms=(score1+score2)/2

    return ms

def compute_H_MSE(gt_H,pre_H):
    return np.linalg.norm(gt_H-pre_H)

# TODO:计算指标部分没有并发加速,计算龟速,可以改
def main(args):
    eval_dataset=EvalDataset(args.dataset,True,output_shape=args.out_size)
    data_loader = DataLoader(eval_dataset,
                                batch_size=1,
                                pin_memory=False,
                                shuffle=False,
                                num_workers=1,
                                worker_init_fn=None,
                                sampler=None) #type:ignore

    #td_kp_detec_params=[('orb',{'nfeatures':1000}),('sift',{'nfeatures':1000})]    
    td_kp_detec_params=[]
    td_kp_detector={x[0]:CVKPDetector(*x) for x in td_kp_detec_params}

    kp_net=KPNet(ckpt_path=args.file)
    matcher=Homographier()

    evaluate_result=defaultdict(dict)
    print('开始收集计算所有图片的结果')
    with torch.no_grad():
        for sample in tqdm(data_loader,total=len(data_loader)):
            kps_info=dict()
            kp1,desc1,kp2,desc2=kp_net(sample['image_t'],sample['warped_image_t'])
            try:
                H=matcher(kp1,desc1,kp2,desc2)
            except:
                set_trace()
            kps_info['kp_net']=(kp1,desc1,kp2,desc2,H)
            sample['homography']=sample['homography'].squeeze().numpy()
            for name,detector in td_kp_detector.items():
                # NOTE: 这里被DataLoader摆了一道,sample里面变成tensor了,后面改
                kp1,desc1,kp2,desc2=detector(sample['image'].squeeze().numpy(),sample['warped_image'].squeeze().numpy())
                H=matcher(kp1,desc1,kp2,desc2)
                kps_info[name]=(kp1,desc1,kp2,desc2,H)

            for detec_name,kp_info in kps_info.items():
                try:
                    _,_,rep,loc_err=compute_repeatability(kp_info[0],kp_info[2],
                                sample['homography'],args.out_size,distance_thresh=args.k)
                except RuntimeWarning:
                    set_trace()
                evaluate_result[detec_name].setdefault('rep',[]).append(rep)
                evaluate_result[detec_name].setdefault('loc_err',[]).append(loc_err)

                # 计算 cor-k
                cor_k=compute_cor_k(args.out_size,sample['homography'],kp_info[4],args.k)
                evaluate_result[detec_name].setdefault('cor_k',[]).append(cor_k)

                # 计算匹配分数
                M_score=compute_M_score(*kp_info[:4],sample['homography'],args.out_size,dis_thr=args.k)
                evaluate_result[detec_name].setdefault('M_score',[]).append(M_score)

                # 计算H矩阵的欧式距离
                distance_H=compute_H_MSE(sample['homography'],kp_info[-1])
                evaluate_result[detec_name].setdefault('dis_H',[]).append(distance_H)
        
        print('开始汇总计算所有结果的平均值')
        for detec_name,result in evaluate_result.items():
            print("=========================================")
            print("{} result".format(detec_name))
            for name,value in result.items():
                print('{}:{}'.format(name,np.mean(value)))

        print("=========================================")
        

if __name__ == '__main__':
    args=parse_args()
    print(args)
    main(args)
