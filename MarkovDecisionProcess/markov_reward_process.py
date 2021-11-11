#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 08:10
# @Author  : zerlinwang
# @File    : markov_reward_process.py
# @Version : 1.0

import numpy as np

P = np.array(
    [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
)

rewards = [-1, -2, -2, 10, 1, 0]

gamma = 0.5


def compute_return(start_index, chain, gamma):
    """
    计算马尔可夫奖励过程的回报

    Parameters
    ----------
    start_index: chain中的开始索引
    chain: 序列
    gamma: 回报折扣

    Returns
    -------
    G: 折扣回报

    """
    G = 0
    for i in reversed(range(start_index, len(chain))):  # 因为最新的reward的gamma为1，所以需要reversed
        G = G * gamma + rewards[chain[i]-1]  # chain好像从s1开始，而reward从s0开始，所以减一
        print(G)
    return G


def compute(P, rewards, gamma, states_num):
    """
    利用贝尔曼方程矩阵形式马尔可夫奖励过程（不包含动作）计算解析解

    Parameters
    ----------
    P: 状态转移矩阵
    rewards: 每个状态的奖励（马尔科夫奖励过程没有动作，奖励由状态唯一决定）
    gamma: 奖励折扣
    states_num: 状态数

    Returns
    -------
    value: np.array，每个状态的价值
    """
    rewards = np.array(rewards).reshape((-1, 1))
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


if __name__ == "__main__":
    # chain = [1, 2, 3, 6]
    # start_index = 0
    # G = compute_return(start_index, chain, gamma)
    # print("得到的回报为：%s" % G)

    V = compute(P, rewards, gamma, 6)
    print("MRP中每个状态价值分别为\n", V)
