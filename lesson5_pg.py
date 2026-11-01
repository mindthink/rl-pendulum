"""第 5 课：策略梯度（REINFORCE）。策略是一个小神经网络，输出连续的推力。

运行：.venv/bin/python lesson5_pg.py [--iters 40] [--seed 0]
  每一轮：用当前策略玩 16 个回合 → 算每一步的“优势” → 按梯度改一次网络
输出：
  out/pg_seed<N>.pt          学到的网络（实时窗口：view.py --ctrl pg）
  out/lesson5_seed<N>.png     学习曲线（立住步数、总分、探索幅度 σ）
  屏幕上：和 LQR 比，能救回的最大冲击
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import viz  # noqa: E402,F401
from controllers import linear, lqr_gains  # noqa: E402
from env import simulate  # noqa: E402
from pg import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs("out", exist_ok=True)

print(f"训练 {args.iters} 轮，每轮 16 个回合（随机种子 {args.seed}）")
policy, hist = train(iters=args.iters, batch=16, seed=args.seed, log=2)
torch.save(policy.state_dict(), f"out/pg_seed{args.seed}.pt")
ctrl = lambda s: policy.act(s, deterministic=True)          # 测试时不加随机，只用平均推力


def max_kick(c, sec=6.0):
    lo, hi = 0.0, 10.0
    for _ in range(14):
        mid = (lo + hi) / 2
        ts, _, _ = simulate(c, seconds=sec, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
        lo, hi = (mid, hi) if ts[-1] >= sec - 0.01 else (lo, mid)
    return lo


surv = [simulate(ctrl, seconds=10, rng=np.random.default_rng(100 + i))[0][-1] for i in range(20)]
print(f"测试（不加随机）：20 个开局里立满 10 s 的有 {sum(t >= 9.99 for t in surv)} 个")
print(f"能救回的最大冲击：策略梯度 {max_kick(ctrl):.2f} rad/s，LQR(R=0.1) {max_kick(linear(lqr_gains(r=0.1))):.2f} rad/s")

it, ln, ret, sig = map(np.array, zip(*hist))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(it, ln, "o-"); ax[0].set_title("平均每回合立住多少步（满分 500）"); ax[0].set_xlabel("训练轮数")
ax[1].plot(it, ret, "o-", color="C1"); ax[1].set_title("平均总分（塑形奖励）"); ax[1].set_xlabel("训练轮数")
ax[2].plot(it, sig, "o-", color="C2"); ax[2].set_title("探索幅度 σ（N）"); ax[2].set_xlabel("训练轮数")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"out/lesson5_seed{args.seed}.png", dpi=110)
print(f"已生成 out/lesson5_seed{args.seed}.png 和 out/pg_seed{args.seed}.pt")
