# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '第二问性能.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/15 4:57'
"""
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
import taichi as ti
ti.init(arch=ti.gpu)


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
@ti.func
def 获取容器(容器表,容器编号):
    for 容器 in 容器表:
        if 容器[0]==容器编号:
            return 容器

@ti.func
def 袋装容器_获取最大立方体的三边(容器数据):
        h = 容器数据[2] + 容器数据[4]
        w = 容器数据[3] + 容器数据[4]
        最长边 = (2 - math.sqrt(2)) * h
        次长边 = (2 - math.sqrt(2)) * w
        最短边 = (math.sqrt(2) - 1) * h
        return 最长边, 次长边, 最短边
        pass
@ti.func
def 袋装容器_获取最长边(容器数据, 长, 宽):
        h = 容器数据[2] + 容器数据[4]
        w = 容器数据[3] + 容器数据[4]
        return h - 长

@ti.func
def 袋装容器_根据首个装入物返回尺寸(已装载容器):
        容器信息 =  已装载容器[0]
        长,宽,高 = 已装载容器[1][0][1]
        最长边 = 袋装容器_获取最长边(容器信息,长,宽)
        return 长,宽,最长边

@ti.func
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


@ti.func
def 装载函数操作_装载角能装下(装载角, 新物品):
    装载角尺寸 = -np.sort(-np.array(装载角[1]))
    新物品尺寸 = np.array(新物品[1:])
    return np.all(装载角尺寸 >= 新物品尺寸)

@ti.func
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


@ti.func
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

@ti.func
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

@ti.func
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
    print(耗材表)
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

        return -1
    数量体积比 = sum(总消耗序列)/总体积

    print("体积数量比=",数量体积比)
    return 数量体积比



@ti.func
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


@ti.kernel
def 差分进化():

    # def f(x):
    #     return sum((x - 0.5) ** 2)


    bounds = [(1,400) for i in range(27)]

    for best, fit in list(de(适应度函数, bounds, its=30,
                             #initpop=np.array([ [0.34185236, 4.52230808, 2.28562625, 5.,         0.72258032, 5.,0.] for i in range(10)])

                             )):
        print("Best solution: {}, fitness: {}".format(best, fit))

if __name__ == "__main__":
    差分进化()
    pass