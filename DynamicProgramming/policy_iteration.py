#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2021/11/12 10:32
# @Author     : zerlinwang
# @File       : policy_iteration.py
# @Description: 策略迭代智能体
# @Version    : 1.0

import copy

from cliff_walk import CliffWalkingEnv
from utils import print_agent


class PolicyIteration:
    """Iteration between policy evaluation and policy improvement"""
    def __init__(self, env, theta, gamma):
        """

        Parameters
        ----------
        env: 环境
        theta: 收敛阈值
        gamma: 折扣系数
        """
        self.env = env
        self.s_sum = len(self.env.P)
        self.a_sum = len(self.env.P[0])
        self.v = [0] * self.s_sum
        self.pi = [[1. / self.a_sum] * self.a_sum for i in range(self.s_sum)]
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self):
        """
        利用不动点迭代法计算状态价值函数v
        以Policy为中心进行迭代，v只是起到辅助作用。这点和价值迭代相反
        Returns
        -------
        None
        """
        # ①
        # v_new = [0] * self.env.ncol * self.env.nrow
        counter = 0
        while True:
            # 放两个地方的差距在哪？？？？？？？？？？？？？？？？？？？？
            # 解答：③使得v_new和self.v指向同一个列表，如果不重新让v_new指向一个新列表，则对v_new的修改会影响到self.v。上一个状态价值的修改在本轮中直接影响到其周围的状态的价值，这是和公式不符的
            # 但是实验发现这样反而使得迭代收敛速度加快了——这个是不动点Jacobi迭代法和Gauss-Seidel迭代法的区别（同步价值迭代和异步价值迭代的区别，后者只用维护一个价值列表，但是稳定性差）
            # ②
            v_new = [0] * self.env.ncol * self.env.nrow
            diff = 0
            # 进行一次迭代，借助v_now计算v_new
            for s in range(self.s_sum):
                q_sum = 0
                for a in range(self.a_sum):
                    q_value = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        # 如果动作后到达的下一个状态游戏结束，到达马尔可夫链的末端，虽然游戏设置为回到开始点，但是计算价值时应该设置为0
                        # 可以发现reward在括号里面而不在外面，这是因为这里reward受到S_{t+1}的影响
                        # 周博磊老师的课程作业一开始没讲这个，想了我好久好久，但是以为reward只和S_{t}和A_{t}有关，想了好久为啥不放在外面
                        q_value += p * (reward + self.gamma * self.v[next_state] * (1-done))
                    q_sum += self.pi[s][a] * q_value
                v_new[s] = q_sum
                # 要求所有的差距都小于theta
                diff = max(diff, abs(v_new[s] - self.v[s]))
            # 注意！！！这句话使得v_new和self.v实际上指向同一个列表！
            # ③
            self.v = v_new
            # self.v = copy.copy(v_new)
            counter += 1
            if diff < self.theta:
                break
            # 判断差距是否大于theta，否则停止迭代
        print(f"策略迭代{counter}轮后收敛。")

    def policy_improvement(self):
        """
        策略提升——更新pi一次，选择动作价值函数最大的动作取值作为策略（如果有多个一样的就是均等分布）
        Returns
        -------
        None，主要更新env.pi
        """
        for s in range(self.s_sum):
            q_list = []
            for a in range(self.a_sum):
                q_value = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    q_value += p * (reward + self.gamma * self.v[next_state] * (1-done))
                q_list.append(q_value)
            q_max = max(q_list)
            q_max_num = q_list.count(q_max)
            self.pi[s] = [1/q_max_num if q_list[i] == q_max else 0. for i in range(self.a_sum)]
        print("策略提升完成")

    def policy_iteration(self):
        while True:
            old_pi = copy.deepcopy(self.pi)
            self.policy_evaluation()
            self.policy_improvement()
            if old_pi == self.pi:
                break


if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

