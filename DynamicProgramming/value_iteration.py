#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2021/11/12 21:02
# @Author     : zerlinwang
# @File       : value_iteration.py
# @Description: 价值迭代智能体
# @Version    : 1.0


class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        # 状态个数与动作个数
        self.s_num = len(self.env.P)
        self.a_num = len(self.env.P[0])
        # **最优**状态价值函数——这是与策略迭代不同的点之一
        self.v = [0] * self.s_num
        self.theta = theta
        self.gamma = gamma
        # 初始化为None的列表，每个状态下策略为None
        self.pi = [[] for i in range(self.s_num)]

    def value_iteration(self):
        """
        策略迭代
        Returns
        -------
        None，主要对于最优状态价值函数v进行修改
        """
        counter = 0
        while True:
            diff = 0
            v_new = [0] * self.s_num
            for s in range(self.s_num):
                v_list = []
                for a in range(self.a_num):
                    v_res = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        # 最后的 * (1-done)容易忘啊
                        v_res += p * (reward + self.gamma * self.v[next_state] * (1-done))
                    v_list.append(v_res)
                v_max = max(v_list)
                v_new[s] = v_max
                diff = max(diff, abs(self.v[s] - v_max))
            if diff < self.theta:
                break
            self.v = v_new
            counter += 1
        print(f"价值迭代一共进行{counter}轮")
        self.get_policy()

    def get_policy(self):
        """
        根据最优状态价值函数导出的贪心策略
        Returns
        -------
        None
        """
        for s in range(self.s_num):
            v_list = []
            for a in range(self.a_num):
                v_res = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    v_res += p * (reward + self.gamma * self.v[next_state] * (1-done))
                v_list.append(v_res)
            v_max = max(v_list)
            max_count = v_list.count(v_max)
            # 让相同的最大动作价值均分概率
            self.pi[s] = [1. / max_count if value == v_max else 0 for value in v_list]


if __name__ == "__main__":
    from cliff_walk import CliffWalkingEnv
    from utils import print_agent

    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
