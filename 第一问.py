# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '第一问.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/13 22:50'
"""
import math

import matplotlib.pyplot as plt
import numpy as np

from 公共变量 import *

容器种类列表 = [
        [0, 0, 250, 190, 1],
        [1, 0, 300, 250, 1],
        [2, 0, 400, 330, 1],
        [3, 0, 450, 420, 1],
        [4, 1, 165, 120, 55],
        [5, 1, 200, 140, 70],
        [6, 1, 200, 150, 150],
        [7, 1, 270, 200, 90],
        [8, 1, 300, 200, 170],
]
只用箱装 = 容器种类列表[4:]
只用袋装 = 容器种类列表[:4]
混合装 = 容器种类列表
# 获取耗材数据()
# 待装载物品序列= [
#         [1, 210, 200, 30],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
#         [1, 170, 110, 27],
# ]
订单序列 = 获取订单展开数据()
唯一订单号 = np.unique(订单序列[:, 0])
订单字典 = {}
差分进化迭代消耗 = []
for 订单号 in 唯一订单号:
    订单字典[订单号] = 订单序列[订单序列[:, 0] == 订单号]

class 权重:
    新容器 = 1
    旧容器 = 2
    边占比 = 2
    面积占比 = 1
    体积占比 = 1
    总体积占比 = 1
    优先贴底 = 1

    def __init__(self,权重值表=None):
        self.权重值表 = [self.新容器,self.旧容器,self.边占比,self.面积占比,self.体积占比,self.总体积占比,self.优先贴底]

        if 权重值表 is not None:
            self.新容器= 权重值表[0]
            self.旧容器= 权重值表[1]
            self.边占比= 权重值表[2]
            self.面积占比= 权重值表[3]
            self.体积占比= 权重值表[4]
            self.总体积占比= 权重值表[5]
            self.优先贴底 = 权重值表[6]
            # for 键,值 in enumerate(权重值表):
            #     self.权重值表[键]=值


def 获取权重():
    return

def 获取容器(容器表,容器编号):
    for 容器 in 容器表:
        if 容器[0]==容器编号:
            return 容器


def 袋装容器_获取最大立方体的三边(容器数据):
        h = 容器数据[2] + 容器数据[4]
        w = 容器数据[3] + 容器数据[4]
        最长边 = (2 - math.sqrt(2)) * h
        次长边 = (2 - math.sqrt(2)) * w
        最短边 = (math.sqrt(2) - 1) * h
        return 最长边, 次长边, 最短边
        pass

def 袋装容器_获取最长边(容器数据, 长, 宽):
        h = 容器数据[2] + 容器数据[4]
        w = 容器数据[3] + 容器数据[4]
        return h - 长

def 袋装容器_根据首个装入物返回尺寸(已装载容器):
        容器信息 =  已装载容器[0]
        长,宽,高 = 已装载容器[1][0][1]
        最长边 = 袋装容器_获取最长边(容器信息,长,宽)
        return 长,宽,最长边

def 物品操作_旋转(物品信息, 底部信息):
    if 底部信息 == 0:
        return 物品信息[1], 物品信息[2], 物品信息[3]  # ab
    elif 底部信息 == 1:
        return 物品信息[1], 物品信息[3], 物品信息[2]  # ac
    elif 底部信息 == 2:
        return 物品信息[2], 物品信息[3], 物品信息[1]  # bc
    elif 底部信息 == 3:
        return 物品信息[2], 物品信息[1], 物品信息[3]  # ba
    elif 底部信息 == 4:
        return 物品信息[3], 物品信息[1], 物品信息[2]  # ca
    elif 底部信息 == 5:
        return 物品信息[3], 物品信息[2], 物品信息[1]  # cb
    else:
        raise ValueError("错误的底部信息!")

def 装载函数操作_装载角能装下(装载角, 新物品):
    装载角尺寸 = -np.sort(-np.array(装载角[1]))
    新物品尺寸 = np.array(新物品[1:])
    return np.all(装载角尺寸 >= 新物品尺寸)

def 装载函数操作_评分(容器信息,x边, y边, z边, 角x边, 角y边, 角z边,新旧权重=权重.旧容器,需要面积=True,需要边=True,需要体积=True,权重实例=None):
        权:权重 = 权重实例 if 权重实例 else 权重
        return 新旧权重 * max(
                权.边占比 * max(x边 / 角x边, y边 / 角y边, z边 / 角z边) if 需要边 else 0 ,
                权.面积占比 * max(
                        (权.优先贴底*x边 * y边) / (角x边 * 角y边),
                        (x边 * z边) / (角x边 * 角z边),
                        (y边 * z边) / (角y边 * 角z边)) if 需要面积 else 0,
                权.体积占比 * (x边 * y边 * z边 )/ (角x边 * 角y边 * 角z边) if  需要体积 else 0,
                权.总体积占比 *(x边 * y边 * z边 )/(容器信息[2]*容器信息[3]*容器信息[4]) if 容器信息[1]==1 else 0
                        #权.总体积占比 *(x边 * y边 * z边) / (角x边 * 角y边 * 角z边)
        )

def 装载函数操作_新增装载角(容器,旧装载角,物品三维):
    装载角x =[辅助.坐标加(旧装载角[0], [物品三维[0], 0, 0]),
           [旧装载角[1][0]-物品三维[0],物品三维[1],物品三维[2]], 0]

    装载角y = [辅助.坐标加(旧装载角[0], [0, 物品三维[1], 0]),
            [物品三维[0],旧装载角[1][1]-物品三维[1],物品三维[2]], 1]

    装载角z = [辅助.坐标加(旧装载角[0], [0, 0, 物品三维[2]]),
            [物品三维[0],物品三维[1],旧装载角[1][2]-物品三维[2]], 2]
    不插入装载角列表=[]

    for 新装载角 in [装载角x,装载角y,装载角z]:
        if 0 in 新装载角[1]:
            不插入装载角列表.append(新装载角)
            for 待比较装载角 in 容器[2]:
                待选坐标 = [
                        辅助.坐标加(待比较装载角[0],[待比较装载角[1][0],0,0]),
                        辅助.坐标加(待比较装载角[0], [ 0,待比较装载角[1][1], 0]),
                        辅助.坐标加(待比较装载角[0], [0, 0, 待比较装载角[1][2]])
                ]
                if 新装载角[0] in 待选坐标:
                    新增尺寸 = [
                            新装载角[1][0] if 新装载角[2] == 0 else 0,
                            新装载角[1][1] if 新装载角[2] == 1 else 0,
                            新装载角[1][2] if 新装载角[2] == 2 else 0,
                    ]
                    待比较装载角[1] = 辅助.坐标加(待比较装载角[1],新增尺寸)
    for 新装载角 in [装载角x, 装载角y, 装载角z]:
        if 新装载角 not in 不插入装载角列表:
            容器[2].append(新装载角)

def 装载函数(待装载物品序列, 待选容器表,权重_=None):
    """
    容器的数据格式: 容器元素=[物品编号,物品三维,插入坐标],容器=[本身属性[容器信息],容器元素表[容器元素],装载角表[装载角],容积],
    装载角=[坐标(x,y,z),尺寸(l,h,w),类型(x=0,y=1,z=2)]
    物品信息=[订单号,长,宽,高], 待装载物品序列 = List[物品信息]
    容器信息=[名称(数字),类型(数字),长,宽,高], 容器种类表 = List[容器信息]
    已选容器列表=[容器]
    评分信息=[旋转,坐标,已选容器编号,新容器编号,得分]
    解释:
    评分信息是指一个物品将被如何放置到容器中,
    旋转: 取值0,1,2,3,4,5分别表示长,宽,高三个数据的全排列取前两个作为物品的底面.
    已选容器编号:  如果已选容器编号=-1则考虑创建新容器
    新容器编号: 是指容器的名称编号, 通常为-1,只有需要新建容器时,才会等于指定的名称编号

    """
    权重_ = 权重() if 权重_ is None else 权重_

    已装载容器序列:"" = [] # 已装载容器序列[容器[]]
    for 新物品编号,新物品 in enumerate(待装载物品序列):
        # 评分环节
        评分表 = []  # 包含物品评分与操作的表

        if 已装载容器序列:
            for 容器编号, 容器 in enumerate(已装载容器序列):
                装载角表 = 容器[2]
                for 装载角 in 装载角表:# 容器2 即装载角
                    if 装载函数操作_装载角能装下(装载角, 新物品):
                        角x边, 角y边, 角z边 = 装载角[1]
                        for i in range(6):
                            x边, y边, z边 = 物品操作_旋转(新物品, i)
                            if 角x边 < x边 or 角y边 < y边 or 角z边 < z边:
                                continue
                            评分 = 装载函数操作_评分(容器,x边, y边, z边,角x边, 角y边, 角z边,权重实例=权重_)
                            评分表.append([i,装载角[0],容器编号,-1,评分])
        for _,待选容器 in enumerate(待选容器表):
            待选容器编号=待选容器[0]
            角x边, 角y边, 角z边 = 待选容器[2], 待选容器[3], 待选容器[4]
            if 待选容器[1] == 1:
                for i in range(6):
                    x边, y边, z边 = 物品操作_旋转(新物品, i)
                    if 角x边<x边 or 角y边<y边 or 角z边<z边:
                        continue

                    评分 = 装载函数操作_评分(待选容器,x边, y边, z边, 角x边, 角y边, 角z边,新旧权重=权重_.新容器,权重实例=权重_)
                    评分表.append([i, (0,0,0), -1, 待选容器编号, 评分])
            else:
                容器最长, 容器次长, 容器最短 = 袋装容器_获取最大立方体的三边(待选容器)

                for i in range(6):
                    x边, y边, z边 = 物品操作_旋转(新物品, i)
                    # 超过这些厚度和宽度就无法装载了, 所以直接跳过
                    物品导出最长边 = 袋装容器_获取最长边(待选容器, x边, y边)
                    if  not (角x边+角z边>=x边+z边 and 角y边+角z边>=y边+z边 ) or 物品导出最长边 < z边:
                        continue
                    else:
                        评分 = 装载函数操作_评分(待选容器, x边, y边, z边, x边, y边, 物品导出最长边, 新旧权重=权重_.新容器,
                                       需要边=False,需要面积=False,权重实例=权重_
                                       )
                        # 评分 = 权重_.新容器 * (max(比例得分, 体积得分))
                        评分表.append([i, (0,0,0), -1, 待选容器编号, 评分])
                pass

        if len(评分表)==0:
            for _,待选容器 in enumerate(待选容器表[0:4]):
                if 新物品[1]+新物品[3]<=待选容器[2]+待选容器[4] and 新物品[2]+新物品[3]<=待选容器[3]+待选容器[4]:
                    容器 = [待选容器.copy(), [[新物品编号, 新物品[1:4], (0,0,0)]], [],新物品[1]*新物品[2]*新物品[3]]
                    已装载容器序列.append(容器)
                    break
            continue

        最佳操作 = max(评分表, key=lambda x: x[-1])
        if 最佳操作[-1] == -1:
            raise ValueError("没有最佳分数, 请检查问题")

        旋转,坐标,容器编号,新容器编号,得分 = 最佳操作
        x边, y边, z边 = 物品操作_旋转(新物品, 旋转)
        # 植入环节
        if 容器编号!=-1: # 此时将新物品插入旧的容器中
            容器 = 已装载容器序列[容器编号]
            容器[1].append([新物品编号,(x边, y边, z边),坐标])
            if 容器[0][1]==0:
                容器[3]+= x边*y边*z边
            装载角 = [角 for 角 in 容器[2] if 角[0]==坐标][0]
            容器[2].remove(装载角)
            装载函数操作_新增装载角(容器,装载角,[x边, y边, z边])
        else:# 插入新容器
            容器 = [获取容器(待选容器表,新容器编号).copy(),[[新物品编号,(x边, y边, z边),坐标]],[],0]
            容器[3] += x边*y边*z边 if 容器[0][1]==0 else 容器[0][2]*容器[0][3]*容器[0][4]
            已装载容器序列.append(容器)
            if 容器[0][1]==1:
                装载角 = [[0,0,0],容器[0][2:5],2]
            else:
                装载角 = [[0, 0, 0],[x边, y边, 袋装容器_获取最长边(容器[0], x边, y边)], 2]
            装载函数操作_新增装载角(容器,装载角, [x边, y边, z边])
            pass
    return 已装载容器序列

def plot_cuboid(ax, corner, size):
    """
    绘制长方体
    :param ax: 绘图轴
    :param corner: 长方体的一个角的坐标 (x, y, z)
    :param size: 长方体的尺寸 (dx, dy, dz)
    """
    x, y, z = corner
    dx, dy, dz = size

    vertices = [
            (x, y, z),
            (x + dx, y, z),
            (x + dx, y + dy, z),
            (x, y + dy, z),
            (x, y, z + dz),
            (x + dx, y, z + dz),
            (x + dx, y + dy, z + dz),
            (x, y + dy, z + dz),
    ]

    edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    faces = [
            [vertices[edge[0]], vertices[edge[1]], vertices[edges[edge[1]][1]], vertices[edges[edge[0]][1]]]
            for edge in edges
    ]

    face_collection = Poly3DCollection(faces, edgecolor='k', linewidths=1)
    face_collection.set_facecolor('c')
    face_collection.set_alpha(0.4)
    ax.add_collection3d(face_collection)

def 直观装载图像绘制(结果,索引=0):
    # 初始化图形和3D坐标轴

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 定义多个长方体的角坐标和尺寸

    # 绘制多个长方体
    第2个容器 = 结果[索引]
    plot_cuboid(ax, (0, 0, 0), 第2个容器[0][2:] if 第2个容器[0][1] == 1 else 袋装容器_根据首个装入物返回尺寸(第2个容器))
    for 方块 in 第2个容器[1]:
        print(方块[2], 方块[1])
        plot_cuboid(ax, 方块[2], 方块[1])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴范围
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_zlim(0, 400)

    # 显示图形
    plt.show()

def 用量绘制(总消耗序列):
    # plt.hist(总消耗序列,bins=100,density=True, alpha=0.75, color='blue', label='耗材消耗分布')
    x = np.linspace(1, len(总消耗序列), len(总消耗序列))
    plt.plot(x,总消耗序列, marker='o', linestyle='-', linewidth=2,)
    plt.show()

def 适应度函数(个体=None, 耗材表=混合装):

    # for 容器种类表 in [只用箱装,只用袋装,混合装]:
    总消耗序列 = []
    总订单序列 = []
    总体积 = 0
    总箱子体积=0
    总箱子个数=0
    总袋子个数=0
    总袋子体积=0
    for 物品序列 in 订单字典.values():

        结果 = 装载函数(物品序列,耗材表,权重(个体))
        总消耗序列.append(len(结果))
        总订单序列.append(结果)
        总体积 +=sum([容器[3]/1000000 for 容器 in 结果])
        总箱子个数 +=sum([1 for 容器 in 结果 if 容器[0][1]==1])
        总袋子个数 +=sum([1 for 容器 in 结果 if 容器[0][1]==0])
        总箱子体积 +=sum([容器[3]/1000000 for 容器 in 结果 if 容器[0][1]==1])
        总袋子体积 += sum([容器[3] / 1000000 for 容器 in 结果 if 容器[0][1] == 0])
    # print(sum(总消耗序列),"箱子=",sum([sum([1 for 容器 in 订单 if 容器[0][1]==1]) for 订单 in 总订单序列]),
    #       "袋子=",sum([sum([1 for 容器 in 订单 if 容器[0][1]==0]) for 订单 in 总订单序列]),
    #       )

    总消耗个数 = sum(总消耗序列)

    print(个体,"总消耗个数=",sum(总消耗序列),"总体积=",总体积,
          "箱子个数=",总箱子个数,"箱子体积=",总箱子体积,
          "袋子个数=",总袋子个数,"袋子体积=",总袋子体积,
          )
    return sum(总消耗序列)
    pass


def 实验装载():
    待装载物品序列= [
            [1, 210, 200, 30],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
            [1, 170, 110, 27],
    ]
    结果 = 装载函数(待装载物品序列, 混合装,权重([3,24,5,26,7,2,9]))
    print(len(结果))

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=10, its=1000,initpop=None):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions) if initpop is None else initpop
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]

def 差分进化(容器列表=混合装):


    # def f(x):
    #     return sum((x - 0.5) ** 2)


    bounds = [(0,5) for i in 权重().权重值表]

    for best, fit in list(de(lambda x:适应度函数(x, 容器列表), bounds, its=30,
                             #initpop=np.array([ [0.34185236, 4.52230808, 2.28562625, 5.,         0.72258032, 5.,0.] for i in range(10)])

                             )):
        print("Best solution: {}, fitness: {}".format(best, fit))


    # print(结果)
    # print(len(结果))
    # 直观装载图像绘制(结果)

if __name__ == "__main__":
    p = np.trunc(np.reshape([313, 109, 400, 1, 202,
                             290, 214, 42, 131, 208,
                             391, 349, 149, 400, 156,
                             23, 36, 1, 1, 194,
                             1, 1, 74, 177, 34,
                             1, 358, ], (9, 3)))
    耗材前半部分 = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 1],
            [5, 1],
            [6, 1],
            [7, 1],
            [8, 1],
    ])
    耗材表 = np.hstack((耗材前半部分, p))
    结果 = 装载函数(订单字典[2264],耗材表,权重_=权重([0.54901561, 4.80096797, 2.352933,   5.,         0.,         5. , 0.12909913]))
    for i in range(len(结果)):
        直观装载图像绘制(结果,索引=i)

    # 差分进化()
    # 实验装载()

    pass
