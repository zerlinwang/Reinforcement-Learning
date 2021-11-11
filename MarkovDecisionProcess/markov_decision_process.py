#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 10:32
# @Author  : zerlinwang
# @File    : markov_decision_process.py
# @Version : 1.0

import numpy as np
from markov_reward_process import compute

S = ["S1", "S2", "S3", "S4", "S5"]  # 状态集合
A = ["保持S1", "前往S1", "前往S2", "前往S3", "前往S4", "前往S5", "概率前往"]    # 动作集合
# 状态转移函数
P = {
    "S1-保持S1-S1":1.0, "S1-前往S2-S2":1.0,
    "S2-前往S1-S1":1.0, "S2-前往S3-S3":1.0,
    "S3-前往S4-S4":1.0, "S3-前往S5-S5":1.0,
    "S4-前往S5-S5":1.0, "S4-概率前往-S2":0.2,
    "S4-概率前往-S3":0.4, "S4-概率前往-S4":0.4,
}
# 奖励函数
R = {
    "S1-保持S1":-1, "S1-前往S2":0,
    "S2-前往S1":-1, "S2-前往S3":-2,
    "S3-前往S4":-2, "S3-前往S5":0,
    "S4-前往S5":10, "S4-概率前往":1,
}
gamma = 0.5 # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1，随即策略
Pi_1 = {
    "S1-保持S1": 0.5, "S1-前往S2": 0.5,
    "S2-前往S1": 0.5, "S2-前往S3": 0.5,
    "S3-前往S4": 0.5, "S3-前往S5": 0.5,
    "S4-前往S5": 0.5, "S4-概率前往": 0.5,
}

# 策略2
Pi_2 = {
    "S1-保持S1":0.6, "S1-前往S2":0.4,
    "S2-前往S1":0.3, "S2-前往S3":0.7,
    "S3-前往S4":0.5, "S3-前往S5":0.5,
    "S4-前往S5":0.1, "S4-概率前往":0.9,
}


def join(str1, str2):
    return str1 + '-' + str2

# 根据策略1将P和R进行转化——MDP转化为MRP，消除动作A
# 转化后的MRP状态转移矩阵
P_from_mdp_to_mrp = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])

R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

if __name__ == "__main__":
    V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
    print("MDP中每个状态价值分别为\n", V)
