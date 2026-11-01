"""模型预测控制（model predictive control, MPC）。

每个控制周期（20 ms）：
  1. 用线性模型 x[k+1] = Ad·x[k] + Bd·F[k] 预测未来 N 步（默认 N = 50，即 1 秒）
  2. 找出让代价 Σ(xᵀQx + R·F²) + 终端代价 x_Nᵀ·P·x_N 最小的推力序列 F[0..N-1]，
     同时满足约束（这是 LQR 不知道的东西）：|F| ≤ 10 N；可选 |x| ≤ x_limit、|θ| ≤ theta_limit（失败线）
  3. 只执行 F[0]，下一个周期用新测到的状态重来（滚动时域，receding horizon）

终端代价用 LQR 的 P：预测窗口之外的部分，按“以后照 LQR 做要付多少”来估算，这样 N 不用很长也能稳定。
优化问题是二次规划（QP），用 OSQP 求解。把未来所有状态都用 F 序列表示出来（“压缩”形式），变量只有 N 个推力。
"""
import numpy as np
import osqp
from scipy import sparse

from controllers import dlqr
from linear import continuous, discretize


class MPC:
    def __init__(self, q=(1.0, 1.0, 10.0, 1.0), r=0.1, horizon=50, f_max=10.0, dt=0.02, x_limit=None, theta_limit=None):
        Ac, Bc = continuous()
        A, B = discretize(Ac, Bc, dt)
        n, N = 4, horizon
        Q = np.diag(q); R = r
        _, P = dlqr(A, B, Q, np.array([[r]]))
        # 预测：X = Sx·x0 + Su·U，X 是第 1..N 步的状态叠起来（4N 维），U 是 N 个推力
        Sx = np.zeros((n * N, n)); Su = np.zeros((n * N, N))
        Ak = np.eye(n)
        for k in range(N):
            Ak = A @ Ak
            Sx[n * k:n * (k + 1)] = Ak
            for j in range(k + 1):
                Su[n * k:n * (k + 1), j] = (np.linalg.matrix_power(A, k - j) @ B).ravel()
        Qbar = np.kron(np.eye(N), Q); Qbar[-n:, -n:] = P          # 最后一步用终端代价 P
        H = Su.T @ Qbar @ Su + R * np.eye(N)
        self.F_lin = Su.T @ Qbar @ Sx                              # 代价里的一次项系数 = F_lin·x0
        # 约束：每一步 −f_max ≤ F ≤ f_max；可选：每一步小车位置 |x| ≤ x_limit
        Arows, lo, hi = [np.eye(N)], [-f_max * np.ones(N)], [f_max * np.ones(N)]
        # 状态约束：小车位置（状态第 0 个）、杆角度（状态第 2 个）不能越过失败线
        self.Sx_state = []
        for idx, lim in ((0, x_limit), (2, theta_limit)):
            if lim is not None:
                Arows.append(Su[idx::n]); self.Sx_state.append(Sx[idx::n])
                lo.append(-lim * np.ones(N)); hi.append(lim * np.ones(N))
        self.Acon = np.vstack(Arows); self.lo0 = np.concatenate(lo); self.hi0 = np.concatenate(hi)
        self.N = N
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(H)), q=np.zeros(N), A=sparse.csc_matrix(self.Acon),
                          l=self.lo0, u=self.hi0, verbose=False, warm_starting=True, polishing=False,
                          eps_abs=1e-6, eps_rel=1e-6, max_iter=4000)
        self.last_plan = np.zeros(N)

    def __call__(self, s):
        s = np.asarray(s, float)
        lo, hi = self.lo0.copy(), self.hi0.copy()
        for i, Sxi in enumerate(self.Sx_state):   # 状态约束要减去“不推时自己会走到哪”
            free = Sxi @ s
            sl = slice(self.N * (i + 1), self.N * (i + 2))
            lo[sl] -= free; hi[sl] -= free
        self.solver.update(q=self.F_lin @ s, l=lo, u=hi)
        res = self.solver.solve()
        if res.x is None or res.info.status_val not in (1, 2):   # 求解失败：沿用上一次计划的下一步
            self.last_plan = np.r_[self.last_plan[1:], 0.0]
        else:
            self.last_plan = res.x
        return float(self.last_plan[0])
