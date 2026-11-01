"""第 1 课：在 MuJoCo 里搭一个倒立摆，并确认它的物理是对的。

运行：.venv/bin/python lesson1_build_sim.py
做三件事：
  1. 打印模型：质量、步长、控制频率
  2. 核对物理：同样的初始状态和推力，MuJoCo 和手写运动方程（sim.py）算出来的轨迹差多少
  3. 开环测试：不看状态地推（不推、一直推、来回推），录视频（out/lesson1_open_loop.mp4）和曲线（.png）
"""
import os

import mujoco
import numpy as np

from env import CartPoleMJ, simulate
from sim import CartPole
from viz import plot, video

os.makedirs("out", exist_ok=True)

# ---- 1. 看模型 ----
env = CartPoleMJ(); m = env.model
print(f"物理步长 {m.opt.timestep * 1000:.0f} ms，控制周期 {env.dt * 1000:.0f} ms（{1 / env.dt:.0f} Hz），推力上限 ±{env.f_max:.0f} N")
for b in ("cart", "pole"):
    print(f"  {b:<5} 质量 {m.body(b).mass[0]:.2f} kg")

# ---- 2. 核对：MuJoCo vs 手写方程（sim.py），同一组随机推力 ----
# 模型里杆的质量属性已经按教科书的均匀细杆写死，所以两边的物理是同一个。
# 剩下的差别只来自手写方程的积分步长：20 ms 一步的欧拉积分（gymnasium 就是这么算的）误差很大，步长越细越接近 MuJoCo。
from sim import Params
for sub, label in ((1, "手写方程，20 ms 欧拉积分"), (40, "手写方程，0.5 ms 欧拉积分"), (400, "手写方程，0.05 ms 欧拉积分")):
    rng = np.random.default_rng(0); worst_deg = 0.0
    for trial in range(20):
        s0 = rng.uniform(-0.05, 0.05, 4)
        mj, eq = CartPoleMJ(), CartPole(p=Params(dt=0.02 / sub))
        mj.reset(s0); eq.reset(s0)
        for k in range(25):                  # 0.5 秒，杆还没倒
            f = rng.uniform(-10, 10)
            a, _ = mj.step(f)
            for _ in range(sub):
                b, _ = eq.step(f)
            worst_deg = max(worst_deg, np.degrees(abs(a[2] - b[2])))
    print(f"MuJoCo vs {label}：20 次 × 0.5 s 随机推，杆角度最大相差 {worst_deg:.2f}°")

# ---- 3. 开环：不看状态的推法 ----
start = [0.0, 0.0, np.radians(1.0), 0.0]      # 杆只偏 1°
k = {"n": 0}
def back_and_forth(s):
    k["n"] += 1
    return 5.0 if ((k["n"] - 1) // 25) % 2 == 0 else -5.0   # 每 0.5 s 换向
runs = []
for title, ctrl in [("不推（F = 0）", lambda s: 0.0), ("一直向右推 3 N", lambda s: 3.0),
                    ("来回推 ±5 N（每 0.5 s 换向）", back_and_forth)]:
    ts, st, fs = simulate(ctrl, seconds=3, state0=start, stop_on_fail=False)
    runs.append((title, ts, st, fs))
    fail_t = next((t for t, s in zip(ts, st) if abs(s[2]) > np.radians(12) or abs(s[0]) > 2.4), None)
    print(f"{title:<16} 杆从 1° 开始，" + (f"{fail_t:.2f} s 后失败" if fail_t else "3 s 内没失败"))
video(runs, "out/lesson1_open_loop.mp4")
plot(runs, "out/lesson1_open_loop.png", "第 1 课：不看状态的推法（开环），杆都会倒")
print("已生成 out/lesson1_open_loop.mp4 和 out/lesson1_open_loop.png")
