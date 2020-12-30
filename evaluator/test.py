# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2020年 12月 30日 星期三 16:52:46 CST
@Description: 用来测试推理结果是否正确,目标evaluator中kp-net推理出来的结果似乎不太对,这里验证以下原始推理接口出来的结果对不对 
'''

import context

import numpy as np

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


def compute_H_MSE(gt_H,pre_H):
    return np.linalg.norm(gt_H-pre_H)

