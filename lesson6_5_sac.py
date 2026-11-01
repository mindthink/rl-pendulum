"""第 6.5 课：SAC。连续推力 + 旧数据反复用。和 PPO 比“学会要多少步数据”。

运行：.venv/bin/python lesson6_5_sac.py [--seed 0] [--steps 30000]   （这台 Mac 上约 8 分钟：每一步都要更新 3 个网络）
输出：out/sac_seed<N>.pt（实时窗口：view.py --ctrl sac）、out/lesson6_5_seed<N>.png（学习曲线和熵权重 α）
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import viz  # noqa: E402,F401
from env import simulate  # noqa: E402
from sac import train  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--target-entropy", type=float, default=-1.0, help="标准值 −1；调到 0.5 探索保留得更多，更稳、更抗推，但学得稍慢")
args = ap.parse_args(); os.makedirs("out", exist_ok=True)
print(f"SAC：训练 {args.steps} 步（随机种子 {args.seed}）")
actor, hist = train(total_steps=args.steps, seed=args.seed, log=2500, target_entropy=args.target_entropy)
tag = f"seed{args.seed}" + ("" if args.target_entropy == -1.0 else f"_te{args.target_entropy:g}")
torch.save(actor.state_dict(), f"out/sac_{tag}.pt")
c = lambda s: actor.act(s, deterministic=True)
surv = sum(simulate(c, seconds=10, rng=np.random.default_rng(100 + i))[0][-1] >= 9.99 for i in range(20))
ts, st, _ = simulate(c, seconds=30, state0=[0, 0, np.radians(2), 0])
lo, hi = 0.0, 10.0
for _ in range(13):
    mid = (lo + hi) / 2
    tt, _, _ = simulate(c, seconds=6, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
    lo, hi = (mid, hi) if tt[-1] >= 5.99 else (lo, mid)
first = next((x[0] for x in hist if x[1] >= 450), None)
print(f"第一次到 450 步以上：第 {first} 步；测试：20 个开局立满 10 s 的有 {surv} 个；30 s："
      f"{'没失败' if ts[-1] >= 29.99 else '%.1f s 失败' % ts[-1]}；能救回的最大冲击 {lo:.2f} rad/s")
t, ln, al = map(np.array, zip(*hist))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(t, ln, "o-"); ax[0].set_title("最近 10 回合平均立住步数（满分 500）"); ax[0].set_xlabel("训练步数")
ax[1].plot(t, al, "o-", color="C2"); ax[1].set_title("熵权重 α（越小 = 越不鼓励随机探索）"); ax[1].set_xlabel("训练步数")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"out/lesson6_5_{tag}.png", dpi=110)
print(f"已生成 out/lesson6_5_{tag}.png 和 out/sac_{tag}.pt")
