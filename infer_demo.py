# -*- coding: utf-8 -*-

# @Description: 
# @Author: CaptainHu
# @Date: 2020-12-22 14:50:00
# @LastEditors: CaptainHu
from typing import List

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from kp2d.networks.keypoint_net import KeypointNet
from kp2d.utils.image import to_color_normalized
from kp2d.evaluation.descriptor_evaluation import select_k_best

# FIXME:注意这里KeyPointModel岁图片进行缩放了,而实际的算出来的H 应该缩放回去
# H_o=S^(-1)_B H_{current} S_A
# 这里若测试图片不是 resize大小  就需要变换回去

class KeyPointModel(object):
    def __init__(self,ckpt_path:str,conf_thresh:float=0.3, keep_k_points=1000):
        checkpoint = torch.load(ckpt_path)
        model_args = checkpoint['config']['model']['params']

        # Create and load disp net
        self.keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                do_upsample=model_args['do_upsample'],
                                do_cross=model_args['do_cross'])
        self.keypoint_net.load_state_dict(checkpoint['state_dict'])
        self.keypoint_net = self.keypoint_net.cuda()
        self.keypoint_net.eval()
        print('Loaded KeypointNet from {}'.format(ckpt_path))
        print('KeypointNet params {}'.format(model_args))
        self.keypoint_net.training = False


        self.eval_params=dict(res=(640,480),top_k=1000)

        self.conf_threshold = conf_thresh
        self.keep_k_points= keep_k_points

        self.transforms=transforms.ToTensor()
    
    def read_img(self,img_path,return_inverse=False):
        r'''读图,顺带返回缩放矩阵
            Args:
                img_path: str
                    图片地址
                return_inverse: bool =False
                    指示返回的缩放矩阵是否是逆阵,直接乘逆阵可以使 new->old
            
            Returns: 
                torch.Tensor: 图片的张量形式
                np.ndarray: 缩放矩阵
        '''
        img=cv2.imread(img_path)
        ori_size=(img.shape[1],img.shape[0])
        img_scale_H=np.eye(3)
        if ori_size !=self.eval_params['res']:
            img=cv2.resize(img,self.eval_params['res'])
            img_scale_H=np.divide(np.array(self.eval_params['res']),ori_size)
            if return_inverse:
                img_scale_H=np.diag(np.append(1./img_scale_H,1.))
            else:
                img_scale_H=np.diag(np.append(img_scale_H,1.))
        img_tensor=self.transforms(img).type('torch.FloatTensor')
        img_tensor=img_tensor.unsqueeze(0)
        return img_tensor,img_scale_H

    def scale_point(self,points:np.ndarray,scale_rate:np.ndarray):
        r'''由于图片在进行变换前可能出现图片被缩放,因此这里需要将点座标缩放回去,这样保证单应性矩阵是原始图片的矩阵
            Args:
                points: numpy.ndarray 
                    关键点集,应该是一个 Nx2 的矩阵
                scale_rate: numpy.ndarray
                    缩放矩阵,应该是一个 3x3 的对角阵.
        '''
        points=np.insert(points,2,1,axis=1)
        new_points=points@scale_rate
        return new_points[:,:2]

    def __call__(self,image_path,warped_image_path):
        image_t,img_t_scale_H=self.read_img(image_path,return_inverse=True)
        warped_image_t,warped_img_t_H=self.read_img(warped_image_path,return_inverse=True)
        
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

            keypoints = prob1[:, :2].T
            keypoints = keypoints[::-1]
            prob = prob1[:, 2]
            keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

            warped_keypoints = prob2[:, :2].T
            warped_keypoints = warped_keypoints[::-1]
            warped_prob = prob2[:, 2]
            warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1], warped_prob], axis=-1)
            
            desc = desc1
            warped_desc = desc2
            keypoints,desc = select_k_best(keypoints, desc, self.keep_k_points)
            warped_keypoints, warped_desc = select_k_best(warped_keypoints, warped_desc, self.keep_k_points)
            fix_kps=self.scale_point(keypoints[:,:2],img_t_scale_H)
            fix_wkps=self.scale_point(warped_keypoints[:,:2],warped_img_t_H)
        return fix_kps,desc,fix_wkps,warped_desc

class Homographier(object):
    def __init__(self,):
        self.matcher=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def __call__(self,kp_1,desc1,kp_2,desc2):
        """
            Args:
                kp_1: numpy.ndarray
                    (N,2)
        """
        matches = self.matcher.match(desc1,desc2)

        matches_idx = np.array([m.queryIdx for m in matches])
        m_keypoints =kp_1[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in matches])
        m_warped_keypoints = kp_2[matches_idx, :]
        H, _ = cv2.findHomography(m_keypoints[:, :2],
                                m_warped_keypoints[:, :2], cv2.RANSAC, 3, maxIters=5000)
        return H

if __name__ == '__main__':
    '''
    ckpt=r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/models/kp2d/v4.ckpt'
    print(infer(ckpt,image_list))
    '''
    ckpt=r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/models/kp2d/v4.ckpt'
    image_list=[
        r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/35kV消弧线圈本体/20200908_10_20_08_1599531621344/35kV消弧线圈油位表.jpg',
        r'/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/35kV消弧线圈本体/20201102_15_00_02_1604300402835/35kV消弧线圈油位表.jpg',
    ]
    matcher=Homographier()
    model=KeyPointModel(ckpt)
    result=model(*image_list)
    H=matcher(*result)
    print(H)