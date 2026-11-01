"""传统控制器（classical controllers）。每个控制器都是一个函数：输入状态 [x, ẋ, θ, θ̇]，输出推力 F（N）。

符号约定：F 为正 = 向右推；θ 为正 = 杆顶向右倒。
所有线性控制器都写成 F = k_x·x + k_xd·ẋ + k_th·θ + k_thd·θ̇，用一个 4 维增益向量 K 表示。
"""
import numpy as np

from linear import continuous, discretize


def linear(K):
    K = np.asarray(K, dtype=float)
    return lambda s: float(K @ s)


# 第 2 课用到的几组手调增益（顺序：x, ẋ, θ, θ̇）
P_THETA = [0, 0, 20, 0]          # 只看角度：P 控制
PD_THETA = [0, 0, 40, 8]         # 角度 + 角速度：PD 控制，但不管小车位置
PD_FULL = [1.0, 2.0, 40, 8]      # 再把小车位置和速度也管起来


def dlqr(Ad, Bd, Q, R, iters=5000, tol=1e-12):
    """离散时间 LQR：迭代求解 Riccati 方程，返回最优反馈 u = −K·x 里的 K。"""
    P = Q.copy()
    for _ in range(iters):
        BtP = Bd.T @ P
        K = np.linalg.solve(R + BtP @ Bd, BtP @ Ad)
        P_new = Q + Ad.T @ P @ (Ad - Bd @ K)
        if np.abs(P_new - P).max() < tol:
            P = P_new
            break
        P = P_new
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def lqr_gains(q=(1.0, 1.0, 10.0, 1.0), r=0.1, dt=0.02):
    """Q = diag(q) 是“状态偏离多少算多糟”，R = r 是“用力多少算多贵”。返回 F = K·s 形式的 4 个增益。"""
    Ac, Bc = continuous()
    Ad, Bd = discretize(Ac, Bc, dt)
    K, _ = dlqr(Ad, Bd, np.diag(q), np.array([[r]]))
    return (-K).ravel()          # LQR 的约定是 u = −K·x；我们写成 F = K·s，所以取负号
