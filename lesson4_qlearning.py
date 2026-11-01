"""第 4 课：表格 Q-learning。

运行：.venv/bin/python lesson4_qlearning.py [--seed 0] [--episodes 3000]
  训练：边玩边更新 Q 表（36 个格子 × 2 个动作）
  测试：只用 Q 最大的动作，20 个随机开局，看平均立住多少步
输出：
  out/q_table_seed<N>.npy    学到的 Q 表（实时窗口可以加载：view.py --ctrl qtable --qfile ...）
  out/lesson4_seed<N>.png     左：学习曲线；中：学到的策略（每个格子往哪推）；右：每个格子的价值 max Q
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import viz  # noqa: E402,F401
from env import CartPoleMJ  # noqa: E402
from qlearn import BINS, greedy_policy, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--episodes", type=int, default=3000)
args = ap.parse_args()
os.makedirs("out", exist_ok=True)

print(f"训练 {args.episodes} 回合（随机种子 {args.seed}）")
Q, disc, lengths = train(args.episodes, seed=args.seed, log=500)
np.save(f"out/q_table_seed{args.seed}.npy", Q)

pol = greedy_policy(Q, disc); env = CartPoleMJ(); rng = np.random.default_rng(5); res = []
for i in range(20):
    s = env.reset(rng=rng); n = 0
    for _ in range(500):
        s, f = env.step(pol(s)); n += 1
        if f:
            break
    res.append(n)
print(f"测试（不再探索）：20 个开局平均立住 {np.mean(res):.0f} 步，满分 500 的有 {sum(r == 500 for r in res)} 个")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].plot(lengths, lw=0.4, alpha=0.4); ax[0].plot(np.convolve(lengths, np.ones(50) / 50, mode="valid"), lw=2, label="50 回合滑动平均")
ax[0].set_title("学习曲线：每回合立住多少步"); ax[0].set_xlabel("回合"); ax[0].grid(alpha=0.3); ax[0].legend()
q2 = Q[0, 0]                                   # 只看杆的两维：θ × θ̇
nth, nthd = q2.shape[:2]
th_edges = np.degrees(np.linspace(-BINS["theta"][1], BINS["theta"][1], nth + 1))
thd_edges = np.degrees(np.linspace(-BINS["theta_dot"][1], BINS["theta_dot"][1], nthd + 1))
act = np.where((q2 == 0).all(-1), np.nan, q2.argmax(-1) * 2 - 1.0)     # −1 向左，+1 向右，没去过的格子留白
im = ax[1].pcolormesh(th_edges, thd_edges, act.T, cmap="coolwarm", vmin=-1, vmax=1, edgecolors="w", lw=1)
tt = np.linspace(th_edges[0], th_edges[-1], 50)
ax[1].plot(tt, -tt / 0.2, "k--", lw=1.5, label="PD 的分界线：θ + 0.2·θ̇ = 0")
ax[1].set_ylim(thd_edges[0], thd_edges[-1]); ax[1].legend(fontsize=9, loc="lower left")
ax[1].set_title("学到的策略：红 = 向右推，蓝 = 向左推"); ax[1].set_xlabel("杆角度 θ（°）"); ax[1].set_ylabel("杆角速度 θ̇（°/s）")
v = np.where((q2 == 0).all(-1), np.nan, q2.max(-1))
im2 = ax[2].pcolormesh(th_edges, thd_edges, v.T, cmap="viridis", edgecolors="w", lw=1); fig.colorbar(im2, ax=ax[2])
ax[2].set_title("每个格子的价值 max Q（越亮 = 以后能拿的分越多）"); ax[2].set_xlabel("杆角度 θ（°）"); ax[2].set_ylabel("杆角速度 θ̇（°/s）")
fig.tight_layout(); fig.savefig(f"out/lesson4_seed{args.seed}.png", dpi=110)
print(f"已生成 out/lesson4_seed{args.seed}.png 和 out/q_table_seed{args.seed}.npy")
