# -*- encoding: utf-8 -*-
'''
Filename         :multi_arm_bandit.py
Description      :老虎机生成类
Time             :2021/11/10 17:12:44
Author           :zerlinwang
Version          :1.0
'''


import numpy as np


class BernoulliBandit:
    """伯努利多臂老虎机，输入K为拉杆个数"""
    def __init__(self, K) -> None:
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = np.max(self.probs)
        self.K = K

    def step(self, k):
        """返回获奖1或未获奖0"""
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    np.random.seed(1)
    K = 10
    bandit_10_arm = BernoulliBandit(K)
    print("随机生成了一个%d臂伯努利老虎机" % K)
    print("获得奖励概率最大的拉杆为%d号，其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))



    