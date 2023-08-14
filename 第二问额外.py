# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '第二问额外.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/15 3:57'
"""
from 第一问 import 只用箱装,只用袋装,订单字典,装载函数,权重,适应度函数,混合装
from 公共变量 import *
#fitness: 0.12798918594433248
[
        ]

def 适应度函数2():
    p = np.trunc(np.reshape(np.array([
            213.50761737, 1., 400,
            386.35330893, 254.29762545, 188.20997538,
            177.88967421, 343.56472848, 185.1473349, 239.92645962,
            301.79973593, 22.00959277, 43.42259817, 118.6268547, 398.98739499,
            133.36778047, 25.92860841, 174.657633, 191.54435006, 163.97234179,
            58.26172976, 96.18762521, 180.09253517, 126.85678997, 178.66020262,
            221.56787028, 228.52837901
    ]), (9, 3)))
    print(p)
    耗材前半部分=np.array([
            [0,0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 1],
            [5, 1],
            [6, 1],
            [7, 1],
            [8, 1],
    ])
    耗材表 = np.hstack((耗材前半部分,p))
    print(耗材表)
    # print(耗材表)
    # 袋装  [0.,         0.78404743, 1.72001252, 4.07501294, 5.         0., 0.92733388]
    # 箱装 [0.78763425, 5.         ,2.61205767 ,5.         ,3.63201236 ,2.6547866, 0.4176828 ]
    # 混装 [0.54901561, 4.80096797, 2.352933,   5.,         0.,         5. , 0.12909913]
    总消耗序列 = []
    总订单序列 = []
    总体积 = 0
    总箱子体积 = 0
    总箱子个数 = 0
    总袋子个数 = 0
    总袋子体积 = 0
    箱_最大个数 = 12678
    袋_最大个数 = 22498
    混_最大个数 = 14837
    箱_最大体积 = 70956
    袋_最大体积 = 46457
    混_最大体积 = 81824
    for 物品序列 in 订单字典.values():
        结果 = 装载函数(物品序列, 耗材表, 权重([0.54901561, 4.80096797, 2.352933,   5.,         0.,         5. , 0.12909913]))
        总消耗序列.append(len(结果))
        总订单序列.append(结果)
        总体积 += sum([容器[3] / 1000000 for 容器 in 结果])
        总箱子个数 += sum([1 for 容器 in 结果 if 容器[0][1] == 1])
        总袋子个数 += sum([1 for 容器 in 结果 if 容器[0][1] == 0])
        总箱子体积 += sum([容器[3] / 1000000 for 容器 in 结果 if 容器[0][1] == 1])
        总袋子体积 += sum([容器[3] / 1000000 for 容器 in 结果 if 容器[0][1] == 0])

    if 箱_最大个数 < 总箱子个数 or 袋_最大个数<总袋子个数 or 混_最大个数<sum(总消耗序列) or \
            箱_最大体积<总箱子体积 or 袋_最大体积<总袋子体积 or 混_最大体积<总体积 :

        return 1000
    数量体积比 = sum(总消耗序列)/总体积

    print("消耗总量=",sum(总消耗序列) ,"总体积=",总体积)
    return 数量体积比


def 适应度函数(p):
    p = np.trunc(np.reshape(p, (9, 3)))
    耗材前半部分=np.array([
            [0,0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 1],
            [5, 1],
            [6, 1],
            [7, 1],
            [8, 1],
    ])
    耗材表 = np.hstack((耗材前半部分,p))
    # print(耗材表)
    # 袋装  [0.,         0.78404743, 1.72001252, 4.07501294, 5.         0., 0.92733388]
    # 箱装 [0.78763425, 5.         ,2.61205767 ,5.         ,3.63201236 ,2.6547866, 0.4176828 ]
    # 混装 [0.54901561, 4.80096797, 2.352933,   5.,         0.,         5. , 0.12909913]
    总消耗序列 = []
    总订单序列 = []
    总体积 = 0
    总箱子体积 = 0
    总箱子个数 = 0
    总袋子个数 = 0
    总袋子体积 = 0
    箱_最大个数 = 12678
    袋_最大个数 = 22498
    混_最大个数 = 14837
    箱_最大体积 = 70956
    袋_最大体积 = 46457
    混_最大体积 = 81824
    for 物品序列 in 订单字典.values():
        结果 = 装载函数(物品序列, 耗材表, 权重([0.54901561, 4.80096797, 2.352933,   5.,         0.,         5. , 0.12909913]))
        总消耗序列.append(len(结果))
        总订单序列.append(结果)
        总体积 += sum([容器[3] / 1000000 for 容器 in 结果])
        总箱子个数 += sum([1 for 容器 in 结果 if 容器[0][1] == 1])
        总袋子个数 += sum([1 for 容器 in 结果 if 容器[0][1] == 0])
        总箱子体积 += sum([容器[3] / 1000000 for 容器 in 结果 if 容器[0][1] == 1])
        总袋子体积 += sum([容器[3] / 1000000 for 容器 in 结果 if 容器[0][1] == 0])

    if 箱_最大个数 < 总箱子个数 or 袋_最大个数<总袋子个数 or 混_最大个数<sum(总消耗序列) or \
            箱_最大体积<总箱子体积 or 袋_最大体积<总袋子体积 or 混_最大体积<总体积 :

        return 1000
    数量体积比 = sum(总消耗序列)/总体积

    print("体积数量比=",数量体积比)
    return 数量体积比

def 差分进化():
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


    # def f(x):
    #     return sum((x - 0.5) ** 2)


    bounds = [(1,400) for i in range(27)]

    for best, fit in list(de(lambda x:适应度函数(x), bounds, its=30,
                             #initpop=np.array([ [0.34185236, 4.52230808, 2.28562625, 5.,         0.72258032, 5.,0.] for i in range(10)])

                             )):
        print("Best solution: {}, fitness: {}".format(best, fit))

if __name__ == "__main__":
    适应度函数2()
    pass