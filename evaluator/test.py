# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2020年 12月 30日 星期三 16:52:46 CST
@Description: 用来测试推理结果是否正确,目标evaluator中kp-net推理出来的结果似乎不太对,这里验证以下原始推理接口出来的结果对不对 
'''

import context

import numpy as np

from evaluator.dataset import EvalDataset
from evaluator.main import parse_args

from ipdb import set_trace

def main(args):
    eval_dataset=EvalDataset(args.dataset,True,output_shape=args.out_size)
    for sample in eval_dataset:
        set_trace() 

if __name__=="__main__":
    args=parse_args()
    main(args)

