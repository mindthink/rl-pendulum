"""第 2.5 课：模型预测控制（MPC）和 LQR 对比。

运行：.venv/bin/python lesson2_mpc.py
打印：同样的 Q、R 下，LQR / 只限推力的 MPC / 把失败线也写进约束的 MPC，能救回的最大冲击
输出：out/lesson2e_mpc.mp4/.png  三者在同一个 2.0 rad/s 冲击下的表现
"""
import os

import numpy as np

from controllers import linear, lqr_gains
from env import simulate
from mpc import MPC
from viz import plot, video

os.makedirs("out", exist_ok=True)


def max_kick(make, sec=6.0):
    lo, hi = 0.0, 10.0
    for _ in range(16):
        mid = (lo + hi) / 2
        ts, _, _ = simulate(make(), seconds=sec, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
        lo, hi = (mid, hi) if ts[-1] >= sec - 0.01 else (lo, mid)
    return lo


LIM = dict(theta_limit=np.radians(11), x_limit=2.3)
print(f"{'R':<6}{'LQR':<8}{'MPC 只限推力':<14}{'MPC + 失败线约束'}")
for r in (1.0, 0.1, 0.01):
    print(f"{r:<6}{max_kick(lambda: linear(lqr_gains(r=r))):<8.2f}{max_kick(lambda: MPC(r=r)):<14.2f}"
          f"{max_kick(lambda: MPC(r=r, **LIM)):.2f}   （单位 rad/s，t = 1 s 时给杆的角速度冲击）")

runs = []
for name, make in [("LQR（R = 0.01）", lambda: linear(lqr_gains(r=0.01))),
                   ("MPC 只限推力（R = 0.01）", lambda: MPC(r=0.01)),
                   ("MPC + 失败线约束（R = 0.01）", lambda: MPC(r=0.01, **LIM))]:
    ts, st, fs = simulate(make(), seconds=5, state0=[0, 0, 0, 0], kicks={1.0: (0, 2.0)})
    runs.append((f"{name}，t = 1 s 被推 2.0 rad/s", ts, st, fs))
video(runs, "out/lesson2e_mpc.mp4"); plot(runs, "out/lesson2e_mpc.png", "同一个冲击：LQR vs MPC")
print("已生成 out/lesson2e_mpc.mp4 和 .png")
