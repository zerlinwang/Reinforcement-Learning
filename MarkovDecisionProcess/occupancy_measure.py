#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 17:17
# @Author  : zerlinwang
# @File    : occupancy_measure.py
# @Version : 1.0

import numpy as np


def occupancy(episodes, s, a, timestep_max, gamma):
    """
    计算状态-动作对(s, a)出现的频率，以此来估算策略的占用度量
    Parameters
    ----------
    episodes: list of list，轨迹序列
    s: dict，状态
    a: dict，动作
    timestep_max: int，最大时间步
    gamma: int，折扣系数

    Returns
    -------
    int, 频率
    """
    rho = 0
    total_times = np.zeros(timestep_max)    # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)    # 记录(s_t, a_t) = (s, a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times [i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]

    rho = (1 - gamma) * rho
    return rho


if __name__ == "__main__":

    from monte_carlo_methods import sample
    from markov_decision_process import MDP, Pi_1, Pi_2

    gamma = 0.5
    timestep_max = 1000

    episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
    episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
    rho_1 = occupancy(episodes_1, "S4", "概率前往", timestep_max, gamma)
    rho_2 = occupancy(episodes_2, "S4", "概率前往", timestep_max, gamma)
    print(rho_1, rho_2)
