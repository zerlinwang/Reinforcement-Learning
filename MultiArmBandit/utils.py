# -*- encoding: utf-8 -*-
'''
Filename         :utils.py
Description      :包含工具函数、类的文件
Time             :2021/11/10 17:13:42
Author           :zerlinwang
Version          :1.0
'''


import matplotlib.pyplot as plt

def plot_results(solvers, solver_names):
    """
    对于每个策略结果进行画图可视化
    Arguments
    ---------
    solvers: 列表，包含各个策略。
    solver_names: 列表，包含各个策略的名字

    Returns
    -------
    None
    """
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed_bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()
    