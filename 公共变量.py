# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '公共变量.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/12 16:26'
"""
import pandas as pd
import torch,math
import dataclasses
import numpy as np
from functools import cmp_to_key
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
excel地址= "./附件1-装箱数据.xlsx"
csv订单数据= "./订单数据.csv"
csv耗材数据 = "./耗材数据.csv"
csv耗材数据名称="./耗材数据名称.csv"
csv物品体积数据 = "./物品体积数据.csv"
csv物品数量展开数据 = "./物品数量展开数据.csv"

耗材名称 = ["普通1号袋",
        "普通2号袋",
        "普通3号袋",
        "普通4号袋",
        "普通1号自营纸箱",
        "普通2号自营纸箱",
        "普通3号自营纸箱",
        "普通4号自营纸箱",
        "普通5号自营纸箱"]
耗材类型 = ["袋", "箱"]


def 获取订单数据():
    return np.loadtxt(csv订单数据,delimiter=",",dtype=int)

def 获取耗材数据():
    return np.loadtxt(csv耗材数据, delimiter=",", dtype=int)

def 获取物品体积数据():
    数据 = 获取订单数据()
    数据 = np.sort(np.repeat(数据[:,1] * 数据[:,2] * 数据[:,3] / 1000000, 数据[:,4]))
    return 数据

def 获取耗材体积数据():
    数据 = 获取耗材数据()
    数据 = np.sort(数据[:, 2] * 数据[:, 3] * 数据[:, 4] / 1000000)
    return 数据

def 获取订单展开数据():
    return np.loadtxt(csv物品数量展开数据, delimiter=",", dtype=int)


class 辅助:
    @staticmethod
    def 坐标加(甲, 乙):
        if len(甲)!=len(乙):
            raise ValueError("长度不同无法相加")

        return [a+b for a,b in  zip(甲,乙)]