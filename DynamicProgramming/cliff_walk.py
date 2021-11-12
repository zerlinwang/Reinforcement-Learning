#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2021/11/12 8:26
# @Author     : zerlinwang
# @File       : cliff_walk.py
# @Description: cliff_walk环境
# @Version    : 1.0


class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.P = self.createP() # 状态转移矩阵 P[state][action] = [(p, next_state, reward, done)]，包含下一个状态和奖励

    # 写了很长时间，还是应该画个分类图
    def createP(self):
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 上下左右，action
        P = [[[] for i in range(len(change))] for j in range(self.ncol * self.nrow)]
        # 定义起点状态
        start_state = (self.nrow - 1) * self.ncol
        end_state = self.nrow * self.ncol - 1
        # (i, j) 对应 (x, y)，左上角为(0, 0)，起点为(0, self.nrow - 1)
        for i in range(self.ncol):
            for j in range(self.nrow):
                for a in range(len(change)):
                    current_state = j * self.ncol + i
                    if current_state > start_state:  # 最后一排非起点，要么当前已经是掉下了悬崖，要么是已经到达了终点
                        # 回到起点
                        P[j*self.ncol+i][a] = [(1, start_state, 0, True)]
                        continue
                    # 设计next_state
                    next_x = max(0, min(self.ncol-1, i + change[a][0]))
                    next_y = max(0, min(self.nrow-1, j + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    # 如果到了悬崖或者终点
                    if next_state > start_state:
                        done = True
                        next_state = start_state
                        # 如果是悬崖
                        if i + change[a][0] < self.ncol - 1:
                            reward = -100
                        else:
                            reward = -1
                    # 普通的一步
                    else:
                        done = False
                        reward = -1

                    P[j*self.ncol+i][a] = [(1, next_state, reward, done)]
        return P


class StandardCliffWalkingEnv:
    """ Cliff Walking环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol # 定义环境的宽
        self.nrow = nrow # 定义环境的高
        self.P = self.createP() # 转移矩阵P[state][action] = [(p, next_state, reward, done)]，包含下一个状态和奖励

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)] # 初始化
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 4 种动作, 0:上, 1:下, 2:左, 3:右。原点(0,0)定义在左上角
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:  # 位置在悬崖或者终点，因为无法继续交互，任何动作奖励都为0
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0: # 下一个位置在悬崖或者终点
                        done = True
                        if next_x != self.ncol - 1: # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


if __name__ == "__main__":
    cliff_env = CliffWalkingEnv()
    for i in range(4):
        for j in range(12):
            R = 0
            for a in range(4):
                # info = cliff_env.P[i*12+j][a]
                # _, _, reward, _ = info[0]
                _, _, reward, _ = cliff_env.P[i*12+j][a]
                R += reward
            print("%6s" % R, end=" ")
        print()

