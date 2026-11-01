"""线性模型：在竖直平衡点（θ = 0）附近，把倒立摆近似成 ẋ = A·x + B·F。第 2 课的 LQR 要用它。

状态顺序和 env.py 一致：[x, ẋ, θ, θ̇]。
  continuous()       教科书推导的连续时间 A、B
  discretize(A,B,T)  变成每 T 秒一步的离散模型 x[k+1] = Ad·x[k] + Bd·F[k]（推力在 T 内保持不变）
  fd_linearize(env)  直接对 MuJoCo 环境做有限差分，量出 20 ms 一步的 Ad、Bd，用来核对上面两步
"""
import numpy as np

M_C, M_P, L_HALF, G = 1.0, 0.1, 0.5, 9.8


def continuous(m_c=M_C, m_p=M_P, l=L_HALF, g=G):
    mt = m_c + m_p
    d = l * (4 / 3 - m_p / mt)
    A = np.zeros((4, 4)); B = np.zeros((4, 1))
    A[0, 1] = 1.0; A[2, 3] = 1.0
    A[3, 2] = g / d                          # θ̈ = (g/d)·θ − F/(mt·d)
    A[1, 2] = -m_p * l * g / (mt * d)        # ẍ = −(m_p·l·g)/(mt·d)·θ + (...)·F
    B[3, 0] = -1 / (mt * d)
    B[1, 0] = 1 / mt + m_p * l / (mt ** 2 * d)
    return A, B


def expm(M, terms=30):
    """矩阵指数（泰勒级数，先缩放再平方，对这么小的矩阵足够精确）。"""
    k = max(0, int(np.ceil(np.log2(max(np.abs(M).sum(), 1e-12)))) + 1)
    Ms = M / 2 ** k
    out = np.eye(len(M)); term = np.eye(len(M))
    for i in range(1, terms):
        term = term @ Ms / i; out = out + term
    for _ in range(k):
        out = out @ out
    return out


def discretize(A, B, T):
    n, m = B.shape
    big = np.zeros((n + m, n + m)); big[:n, :n] = A; big[:n, n:] = B
    E = expm(big * T)
    return E[:n, :n], E[:n, n:]


def fd_linearize(env, eps=1e-6):
    """在 θ = 0、静止、F = 0 附近，用中心差分量出一个控制周期的 Ad、Bd。"""
    n = 4
    Ad = np.zeros((n, n)); Bd = np.zeros((n, 1))
    for i in range(n):
        d = np.zeros(n); d[i] = eps
        env.reset(d); sp, _ = env.step(0.0)
        env.reset(-d); sm, _ = env.step(0.0)
        Ad[:, i] = (sp - sm) / (2 * eps)
    env.reset(np.zeros(n)); sp, _ = env.step(eps)
    env.reset(np.zeros(n)); sm, _ = env.step(-eps)
    Bd[:, 0] = (sp - sm) / (2 * eps)
    return Ad, Bd
