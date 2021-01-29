# -*- coding: utf-8 -*-
'''
@Author: CaptainHu
@Date: 2021年 01月 29日 星期五 09:46:00 CST
@Description: 用于绘制训练过程中eval的结果
'''
"""
使用:
使用前使用rg或者grep搜出关键字的行  在跑这脚本
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='绘图脚本')
    parser.add_argument('-f',"--file",type=str)
    args=parser.parse_args()
    return args
    

def parse_log(log_path):
    repeatability=[]
    loc_error=[]
    cor1=[]
    cor3=[]
    cor5=[]
    m_score=[]
    with open(log_path, 'r') as f:
        for line in f.readlines():
            if line.startswith("Repeatability"):
                repeatability.append(float(line.strip().split(" ")[-1]))
            elif line.startswith("Localization"):
                loc_error.append(float(line.strip().split(" ")[-1]))
            elif line.startswith("Correctness d1"):
                cor1.append(float(line.strip().split(" ")[-1]))
            elif line.startswith("Correctness d3"):
                cor3.append(float(line.strip().split(" ")[-1]))
            elif line.startswith("Correctness d5"):
                cor5.append(float(line.strip().split(" ")[-1]))
            elif line.startswith("MScore"):
                m_score.append(float(line.strip().split(" ")[-1]))
            else:
                continue
    return repeatability,loc_error,cor1,cor3,cor5,m_score

def draw_plot(ax,data,color,legend):
    x=np.arange(len(data))
    y=np.array(data)
    ax.plot(x,y,color,label=legend)
    
def main(log_path):
    rep,loc_err,c1,c3,c5,m_s=parse_log(log_path)
    
    fig=plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    draw_plot(ax,rep,'r','rep')
    draw_plot(ax,loc_err,'tan','loc_err')
    draw_plot(ax,c1,'b','cor1')
    draw_plot(ax,c3,'c','cor3')
    draw_plot(ax,c5,'y','cor5')
    draw_plot(ax,m_s,'g','m_score')
    ax.legend(loc='best')
    plt.show()

if __name__=="__main__":
    args=parse_args()
    main(args.file)
    

