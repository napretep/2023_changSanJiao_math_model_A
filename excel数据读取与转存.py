# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = 'excel数据读取与转存.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/12 12:57'
"""
import numpy as np

from 公共变量 import *



def 处理Excel数据(path):
    #读取
    原始数据集 = pd.read_excel(path,sheet_name =None)
    return 原始数据集
    # 订单数据集 = 原始数据集["订单数据"]
    # 耗材数据集 = 原始数据集["耗材数据"]
    # 保存订单数据(订单数据集)
    # 保存耗材数据(耗材数据集)

def 保存耗材数据(耗材数据集):
    耗材数据_整理 = []

    for 行 in 耗材数据集.values[:,:]:
        名称 = 耗材名称.index(行[0])
        类型 = 耗材类型.index(行[1])
        最长边, 次长边, 最短边 = sorted(行[2:5],reverse=True)
        耗材数据_整理.append([名称,类型,最长边, 次长边, 最短边])

    耗材数据_整理 = np.array(耗材数据_整理,dtype=int)
    np.savetxt(csv耗材数据,耗材数据_整理,delimiter=",",fmt="%d")

def 保存订单数据(订单数据集):
    订单数据_整理 = []
    # 排序1, 最长边, 次长边, 最短边
    for 行 in 订单数据集.values[:, :]:
        最长边, 次长边, 最短边 = sorted(行[1:4], reverse=True)
        订单编号, 数量 = 行[0], 行[4]
        订单数据_整理.append([订单编号, 最长边, 次长边, 最短边, 数量])

    订单数据_整理 = 数据排序(数据清洗过滤(np.array(订单数据_整理, dtype=int)))


    np.savetxt(csv订单数据, 订单数据_整理, delimiter=",", fmt="%d")

def 数据清洗过滤(data):


    # 包装箱长宽高
    boxes = np.array([
            [165, 120, 55],
            [200, 140, 70],
            [200, 150, 150],
            [270, 200, 90],
            [300, 200, 170]
    ])

    # 袋型包装长宽高
    bags = np.array([
            [250, 190, 1],
            [300, 250, 1],
            [400, 330, 1],
            [450, 420, 1]
    ])

    # 定义一个函数，返回一个布尔值，指示商品是否可以放入至少一个包装箱
    def can_fit_in_any_box(item, boxes):
        return np.any(np.all(item[1:4] < boxes, axis=1))

    # 定义一个函数，返回一个布尔值，指示商品是否可以放入至少一个袋型包装
    def can_fit_in_any_bag(item, bags):
        return np.any((item[1] + item[3] <= bags[:,0] + bags[:,2]) & (item[2] + item[3] <= bags[:,1] + bags[:,2]))

    # 根据条件筛选出符合要求的商品
    filtered_data = data[np.array([can_fit_in_any_box(item, boxes) | can_fit_in_any_bag(item, bags) for item in data])]
    # filtered_data = data[np.array([can_fit_in_any_box(item, boxes) for item in data])]
    print(filtered_data)


    return filtered_data

def 数据排序(data):

    sorted_indices = np.lexsort((-data[:, 3], -data[:, 2], -data[:, 1], data[:, 0]))

    return data[sorted_indices]

def 保存物品体积数据():
    数据 = 获取物品体积数据()
    np.savetxt(csv物品体积数据,数据,delimiter=",",fmt='%.4f')

def 保存物品数量展开数据():
    数据 = 获取订单数据()
    数据 = np.repeat(数据, 数据[:, 4],axis=0)
    数据 = np.delete(数据,-1,axis=1)
    np.savetxt(csv物品数量展开数据,数据,delimiter=",",fmt='%d')

if __name__ == "__main__":
    保存物品数量展开数据()

    pass