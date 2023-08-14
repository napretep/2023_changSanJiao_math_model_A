# -*- coding: utf-8 -*-
"""
__project_ = 'project'
__file_name__ = '数据可视化.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/5/12 16:26'
"""
import numpy as np

from 公共变量 import *

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
数据 = 获取订单数据()
体积集合 = np.sort(np.repeat(数据[:, 1] * 数据[:, 2] * 数据[:, 3]/1000000, 数据[:, 4]))
kstest = stats.kstest
gamma = stats.gamma

def 绘制商品体积分布图():

    # 假设你的数据存储在一个2D的ndarray中，每一行代表一个订单

    # 计算每个订单中所有产品的体积之和和数量之和

    上界 = np.percentile(体积集合, 0)
    下界 = np.percentile(体积集合, 90)

    print(f"全部商品的体积 90% 集中在区间 ({上界}, {下界})")
    plt.hist(体积集合, bins=100)

    plt.axvline(上界, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(下界, color='green', linestyle='dashed', linewidth=2,label="左侧")
    x_fill = np.linspace(上界, 下界, 100)
    y_fill = np.max(体积集合) - np.min(体积集合)  # 设置遮罩范围以覆盖整个 y 范围

    # 添加黄色透明遮罩
    plt.fill_between(x_fill, 0, 2000, color='yellow', alpha=0.5)

    plt.xlabel("体积-$m^3$")
    plt.ylabel("个数")
    plt.title('体积分布')
    plt.show()


def 绘制商品体积拟合曲线图():
    shape, loc, scale = gamma.fit(体积集合)
    x = np.linspace(np.min(体积集合), np.max(体积集合), 1000)
    pdf_gamma = gamma.pdf(x, shape, loc, scale)

    d_value, p_value = kstest(体积集合, 'gamma', args=(shape, loc, scale))

    print(f"D 值：{d_value:.4f}")
    print(f"p 值：{p_value:.4f}")

    plt.hist(体积集合, bins=100, density=True, alpha=0.75, color='blue', label='体积分布')
    plt.plot(x, pdf_gamma, 'r-', lw=2, label='拟合的伽马分布')
    plt.legend()
    plt.xlabel('体积-$m^3$')
    plt.ylabel('密度/数量')
    plt.title('体积分布与拟合伽马分布')

    # 显示图像
    plt.show()

def 绘制耗材体积分布图():
    数据 = 获取耗材数据()





if __name__ == "__main__":
    # 创建一个简单的曲线
    import numpy as np
    import matplotlib.pyplot as plt
