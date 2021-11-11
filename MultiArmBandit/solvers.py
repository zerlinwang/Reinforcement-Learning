# -*- encoding: utf-8 -*-
'''
Filename         :solvers.py
Description      :各种策略文件
Time             :2021/11/10 17:14:08
Author           :zerlinwang
Version          :1.0
'''


import numpy as np


class Solver:
    """
    多臂老虎机算法基本框架
    
    Q: 为啥可以访问老虎机的最高概率的arm的期望概率来求regret？
    A: regret只是用来可视化算法性能，在run_one_step即决策过程中没有用到best_prob，只有estimate
    """
    def __init__(self, bandit) -> None:
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0.    # The cumulative regret until current step
        self.regrets = []   # 记录每一步的累计懊悔
        self.actions = []   # 记录每一步的动作
    
    def update_regret(self, k):
        self.regret += (self.bandit.best_prob - self.bandit.probs[k])
        self.regrets.append(self.regret)

    def run_one_step(self):
        """根据具体策略返回当前选择的拉杆并更新对每个拉杆的奖励期望估计"""
        return NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.actions.append(k)
            self.counts[k] += 1
            self.update_regret(k)
        

class EpsilonGreedy(Solver):
    """Epsilon贪心算法"""
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)  # 初始化所有拉杆的奖励估计

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机选择一个arm
        else:
            k = np.argmax(self.estimates)   # 选择奖励期望最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class DecayingEpsilonGreedy(Solver):
    """衰减的Epsilon贪心算法"""
    def __init__(self, bandit, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1. / self.total_count:    # epsilon随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class UCB(Solver):
    """UCB算法"""
    def __init__(self, bandit, coef, init_prob=1.0) -> None:
        super().__init__(bandit)
        self.total_num = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_num += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_num) / (2 * (self.counts + 1)))    # 计算上置信界
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += (r - self.estimates[k]) / (self.counts[k] + 1)
        return k


class ThompsonSampling(Solver):
    """汤普森算法"""
    def __init__(self, bandit) -> None:
        super().__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        self._a[k] += r
        self._b[k] += 1 - r
        return k
