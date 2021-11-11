import numpy as np

from multi_arm_bandit import BernoulliBandit
from solvers import *
from utils import plot_results


def epsilon_greedy_train(bandit):

    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print("epsilon贪心算法的懊悔程度为：", epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


def multi_epsilon_greedy_train(bandit):

    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    epsilon_greedy_solver_list = [EpsilonGreedy(
        bandit, epsilon=e) for e in epsilons]
    epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(5000)
    plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


def decaying_epsilon_greedy_train(bandit):

    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
    decaying_epsilon_greedy_solver.run(5000)
    print("epsilon贪心算法的懊悔程度为：", decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])


def upper_confident_bound_train(bandit):
    np.random.seed(1)
    coef = 0.1 # 控制不确定性比重的系数
    UCB_solver = UCB(bandit, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])


def thompthon_sampling_train(bandit):
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])


if __name__ == "__main__":
    np.random.seed(1)
    bandit_10_arm = BernoulliBandit(10)
    # epsilon_greedy_train(bandit_10_arm)
    # multi_epsilon_greedy_train(bandit_10_arm)
    # decaying_epsilon_greedy_train(bandit_10_arm)
    # upper_confident_bound_train(bandit_10_arm)
    thompthon_sampling_train(bandit_10_arm)
