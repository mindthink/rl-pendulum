"""第 6 课：PPO，和第 5 课的 REINFORCE 比。

运行：.venv/bin/python lesson6_ppo.py [--seed 0]
  训练 PPO 60 轮，每轮 2048 步（约 2 分钟），同时把 REINFORCE 也训一遍，按“用了多少步数据”对比
输出：
  out/ppo_seed<N>.pt          学到的策略（实时窗口：view.py --ctrl ppo）
  out/lesson6_seed<N>.png     学习曲线对比、σ 的变化
  屏幕上：两者和 LQR 的抗冲击能力
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import viz  # noqa: E402,F401
import pg  # noqa: E402
import ppo  # noqa: E402
from controllers import linear, lqr_gains  # noqa: E402
from env import simulate  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
os.makedirs("out", exist_ok=True)


def max_kick(c, sec=6.0):
    lo, hi = 0.0, 10.0
    for _ in range(13):
        mid = (lo + hi) / 2
        ts, _, _ = simulate(c, seconds=sec, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
        lo, hi = (mid, hi) if ts[-1] >= sec - 0.01 else (lo, mid)
    return lo


print("PPO：60 轮 × 2048 步")
pol, critic, h_ppo = ppo.train(iters=60, seed=args.seed, log=5)
torch.save(pol.state_dict(), f"out/ppo_seed{args.seed}.pt")
print("REINFORCE（第 5 课）：40 轮 × 16 回合")
pol_pg, h_pg = pg.train(iters=40, batch=16, seed=args.seed, log=10)

ppo_ctrl = lambda s: pol.act(s, deterministic=True)
pg_ctrl = lambda s: pol_pg.act(s, deterministic=True)
print(f"能救回的最大冲击：PPO {max_kick(ppo_ctrl):.2f}，REINFORCE {max_kick(pg_ctrl):.2f}，LQR(R=0.1) {max_kick(linear(lqr_gains(r=0.1))):.2f} rad/s")

steps_pg = np.cumsum([16 * x[1] for x in h_pg])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot([x[1] for x in h_ppo], [x[3] for x in h_ppo], "o-", label="PPO")
ax[0].plot(steps_pg, [x[1] for x in h_pg], "s-", label="REINFORCE")
ax[0].set_xlabel("累计用了多少步数据"); ax[0].set_title("平均每回合立住多少步（满分 500）"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot([x[1] for x in h_ppo], [x[4] for x in h_ppo], "o-", label="PPO")
ax[1].plot(steps_pg, [x[3] for x in h_pg], "s-", label="REINFORCE")
ax[1].set_xlabel("累计用了多少步数据"); ax[1].set_title("探索幅度 σ（N）"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"out/lesson6_seed{args.seed}.png", dpi=110)
print(f"已生成 out/lesson6_seed{args.seed}.png 和 out/ppo_seed{args.seed}.pt")
