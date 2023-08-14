# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '第二问.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/15 1:52'
"""
import numpy as np

from 第一问 import 差分进化,只用箱装,只用袋装,订单字典,装载函数,权重,适应度函数
from 公共变量 import *

# Sphere 函数
def sphere(x):
    return np.sum(x**2)


def fit(p):
    p = np.reshape(p, (9, 3))
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
    体积数量比 = 总体积/sum(总消耗序列)

    print("体积数量比=",体积数量比)
    return 体积数量比



# 粒子群优化算法实现
def PSO(fitness_function, n_variables, n_particles, n_iterations, bounds, w=0.7, c1=1.5, c2=1.5):


    # 初始化粒子位置和速度
    particle_positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_variables))
    particle_velocities = np.random.uniform(-abs(bounds[:, 1] - bounds[:, 0]), abs(bounds[:, 1] - bounds[:, 0]), (n_particles, n_variables))
    print("初始化粒子位置和速度")
    # 初始化粒子的个体最佳位置和群体最佳位置
    particle_best_positions = np.copy(particle_positions)
    particle_best_fitness_values = np.array([fitness_function(pos) for pos in particle_positions])
    global_best_position = particle_best_positions[np.argmin(particle_best_fitness_values)]
    global_best_fitness = np.min(particle_best_fitness_values)
    print("初始化粒子的个体最佳位置和群体最佳位置")
    # 迭代更新粒子位置和速度
    for _ in range(n_iterations):
        # 更新粒子速度
        r1, r2 = np.random.rand(n_particles, n_variables), np.random.rand(n_particles, n_variables)
        cognitive_velocity = c1 * r1 * (particle_best_positions - particle_positions)
        social_velocity = c2 * r2 * (global_best_position - particle_positions)
        particle_velocities = w * particle_velocities + cognitive_velocity + social_velocity

        # 更新粒子位置
        particle_positions = particle_positions + particle_velocities

        # 将粒子位置限制在边界内
        particle_positions = np.clip(particle_positions, bounds[:, 0], bounds[:, 1])

        # 计算新位置的适应度值
        fitness_values = np.array([fitness_function(pos) for pos in particle_positions])

        # 更新粒子的个体最佳位置
        improved_particles = fitness_values < particle_best_fitness_values
        particle_best_positions[improved_particles] = particle_positions[improved_particles]
        particle_best_fitness_values[improved_particles] = fitness_values[improved_particles]

        # 更新群体最佳位置
        if np.min(fitness_values) < global_best_fitness:
            global_best_position = particle_positions[np.argmin(fitness_values)]
            global_best_fitness = np.min(fitness_values)

    return global_best_position, global_best_fitness
# 问题参数设置
bounds = np.tile(np.array([[1, 400], [1, 400], [1, 400]]), (9, 1))  # 重复 9 次以匹配 9 种耗材

n_variables = 9 * 3  # 为 9 种耗材的每个维度创建一个决策变量
n_particles = 50
n_iterations = 100


if __name__ == "__main__":

    pass
    best_consumable_sizes, best_fitness = PSO(fit, n_variables, n_particles, n_iterations, bounds)

    print('Best consumable sizes:')
    print(best_consumable_sizes)
    print('Best fitness:', best_fitness)
    # 差分进化(容器列表=只用箱装)
    # 适应度函数(耗材表=只用袋装)