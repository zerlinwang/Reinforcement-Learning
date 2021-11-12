#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2021/11/12 15:31
# @Author     : zerlinwang
# @File       : utils.py
# @Description: 包含画图函数
# @Version    : 1.0

def print_agent(agent, action_meaning, disaster=[], end=[]):
    """
    打印当前策略每个状态下的价值以及会采取的动作。
    对于打印出来的动作，我们用"^o<o"表示等概率采取向左和向上两种动作，"ooo>"表示在当前状态只采取向右动作。
    Parameters
    ----------
    agent
    action_meaning
    disaster
    end

    Returns
    -------

    """
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ') # 为了输出美观，保持输出6个字符
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster: # 一些特殊的状态，例如Cliff Walking中的悬崖
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end: # 终点
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0. else 'o'
                print(pi_str, end=' ')
        print()
