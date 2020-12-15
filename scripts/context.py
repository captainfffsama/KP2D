# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2020年 12月 15日 星期二 15:40:04 CST
@Description: 把上一层包加入到系统路径让包能找到
'''

import os 
import sys

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
