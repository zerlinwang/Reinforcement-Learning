#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 16:23
# @Author  : zerlinwang
# @File    : monte_carlo_methods.py
# @Version : 1.0

import numpy as np
from markov_decision_process import join


def sample(MDP, Pi, timestep_max, number):
    """
    采样函数，计算马尔科夫序列(episodes)
    Parameters
    ----------
    MDP: 5-tuple，马尔可夫决策过程
    Pi: dict，策略Pi
    timestep_max: int，限制最长时间步
    number: int，采样序列个数

    Returns
    -------
    episodes: 列表，马尔科夫序列
    """
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)] # 从除了s5以外的状态随机初始化作为起点
        while s != "S5" and timestep <= timestep_max:
            timestep += 1
            # 这个动作选择过程 有点迷惑，就不能随机在A中选一个非0的吗？——是等价的，因为有0概率，所以也许这个写法是最方便的。
            rand, temp = np.random.rand(), 0
            # 在s根据策略Pi选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))   # 将一个轨迹放入episode中
            s = s_next
        episodes.append(episode)
    return episodes


def MC(episodes, V, N, gamma):
    """
    蒙特卡洛法对所有序列计算所有状态的价值
    根据大数定律，N(s)->∞，有V(s)->V^{pi}(s)
    Parameters
    ----------
    episodes: list of list，轨迹序列
    V: dict，储存每个状态的价值，初始化为0
    N: dict，储存每个状态途径的次数，初始化为0
    gamma: float，reward折扣系数

    Returns
    -------
    None，将计算出的值储存在V和N中
    """
    for episode in episodes:
        G = 0
        for i in range(len(episode)-1, -1, -1): # 每条episode都从后面向前进行计算
            (s, a, r, s_next) = episode[i]
            # 序列最后面的状态奖励会乘以最多的折扣系数
            G = r + gamma * G
            N[s] += 1
            # 此时的G刚好是从s出发到最后的奖励，即状态价值V
            V[s] += (G - V[s]) / N[s]   # 增量更新方式，V始终是均值


if __name__ == "__main__":
    from markov_decision_process import MDP, Pi_1

    # episodes = sample(MDP, Pi_1, 20, 5)
    # print('第一条序列\n', episodes[0])
    # print('第二条序列\n', episodes[1])
    # print('第五条序列\n', episodes[4])

    timestep_max = 20
    # 采样1000次，可以自行修改
    episodes = sample(MDP, Pi_1, timestep_max, 1000)
    gamma = 0.5
    V = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0}
    N = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0}
    MC(episodes, V, N, gamma)
    print("使用蒙特卡洛法计算MDP状态价值为\n", V)

    # 没有设置为随机数种子，但结果仍然与教程结果/解析解接近，大数定理/蒙特卡洛方法起作用
