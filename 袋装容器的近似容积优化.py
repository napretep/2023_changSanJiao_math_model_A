# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '袋装容器的近似容积优化.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/13 3:26'
"""
from 公共变量 import *
当前目标容器 = [0,0,250,190] # 获取耗材数据()[0]
def 最大容积(w, h):
    return w**3 * (h / (np.pi * w) - 0.142 * (1 - 10**(-h / w)))


def 适应度(长,宽,高):
    V_ = 长*宽*高
    V = 最大容积(当前目标容器[2]+1,当前目标容器[3]+1)
    return (V-V_)/V


if __name__ == "__main__":

    # 确保 PyTorch 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def fitness_function(params):
        长, 宽, 高 = params
        return 适应度(长, 宽, 高)  # 由于差分进化寻找最小值，我们取负值


    def constraint(params):
        长, 宽, 高 = params
        return 长 > 0 and 宽 > 0 and 高 > 0 and 长 + 高 <= 当前目标容器[2] + 1 and 宽 + 高 <= 当前目标容器[3] + 1


    def differential_evolution(func, bounds, pop_size, max_iter, F=0.8, CR=0.9):
        dimensions = len(bounds)
        lower_bounds, upper_bounds = torch.tensor(bounds).T.to(device)
        pop = (torch.rand(pop_size, dimensions).to(device) * (upper_bounds - lower_bounds) + lower_bounds).to(device)

        # 设置交互式绘图
        plt.ion()
        fig, ax = plt.subplots()
        avg_fitness_values = []

        for i in range(max_iter):
            trial_pop = torch.empty_like(pop)
            for j, x in enumerate(pop):
                # 检查约束条件，如果不满足则跳过
                if not constraint(x.cpu().numpy()):
                    trial_pop[j] = x
                    continue

                a, b, c = pop[torch.randperm(pop_size)[:3]]
                mutant = a + F * (b - c)
                crossover = torch.rand(dimensions).to(device) < CR
                trial = torch.where(crossover, mutant, x)
                trial_pop[j] = trial if func(trial) < func(x) else x
            pop = trial_pop

            # 计算平均适应度值
            fitness_values = torch.tensor([func(x) for x in pop]).to(device)
            best_individual = pop[torch.argmin(torch.tensor([func(x) for x in pop]).to(device))]
            avg_fitness = torch.mean(fitness_values)
            avg_fitness_values.append(avg_fitness.item())

            # 绘制实时进度
            ax.clear()
            ax.plot(range(1, i + 2), avg_fitness_values)
            ax.set_title(f"Iteration {i + 1}, best = {best_individual}, score={avg_fitness.item()}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Average Fitness")
            plt.draw()
            plt.pause(0.01)

        # 关闭交互式绘图
        plt.ioff()

        best_individual = pop[torch.argmin(torch.tensor([func(x) for x in pop]).to(device))]
        return best_individual, func(best_individual)


    # 参数设置
    bounds = [(0, 当前目标容器[2]), (0, 当前目标容器[3]), (0, min(当前目标容器[2], 当前目标容器[3]))]
    pop_size = 20  # 种群大小
    max_iter = 1000  # 最大迭代次数

    # 使用差分进化求解问题
    best_solution, best_fitness = differential_evolution(fitness_function, bounds, pop_size, max_iter)
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {-best_fitness.item()}")  # 我们取适应度函数的负值，所以这里需要取反pass