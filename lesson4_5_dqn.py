"""第 4.5 课：DQN。Q 表换成神经网络，这次连小车位置也看。

运行：.venv/bin/python lesson4_5_dqn.py [--reward shaped|survive] [--no-replay] [--no-target] [--seed 0]
  默认：塑形奖励 + 经验回放 + 目标网络，训练 6 万步（这台 Mac 上约 3 分钟）
输出：out/dqn_<设置>.pt（实时窗口：view.py --ctrl dqn --dqnfile ...）、out/lesson4_5_<设置>.png（学习曲线）
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import viz  # noqa: E402,F401
from dqn import greedy, train  # noqa: E402
from env import simulate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reward", default="shaped", choices=["shaped", "survive"])
ap.add_argument("--no-replay", action="store_true"); ap.add_argument("--no-target", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=60000)
args = ap.parse_args()
tag = f"{args.reward}{'_noreplay' if args.no_replay else ''}{'_notarget' if args.no_target else ''}_seed{args.seed}"
os.makedirs("out", exist_ok=True)

print(f"DQN：{tag}，训练 {args.steps} 步")
q, hist = train(total_steps=args.steps, reward=args.reward, replay=not args.no_replay, target=not args.no_target,
                seed=args.seed, log=2500)
torch.save(q.state_dict(), f"out/dqn_{tag}.pt")
c = greedy(q)
surv = sum(simulate(c, seconds=10, rng=np.random.default_rng(100 + i))[0][-1] >= 9.99 for i in range(20))
ts, st, _ = simulate(c, seconds=30, state0=[0, 0, np.radians(2), 0])
lo, hi = 0.0, 10.0
for _ in range(13):
    mid = (lo + hi) / 2
    tt, _, _ = simulate(c, seconds=6, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
    lo, hi = (mid, hi) if tt[-1] >= 5.99 else (lo, mid)
print(f"测试：20 个开局立满 10 s 的有 {surv} 个；从 2° 开始跑 30 s："
      f"{'没失败' if ts[-1] >= 29.99 else '%.1f s 失败' % ts[-1]}，小车最远 {np.abs(st[:, 0]).max():.2f} m；"
      f"能救回的最大冲击 {lo:.2f} rad/s")
t, ln, eps = map(np.array, zip(*hist))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, ln, "o-", label="最近 20 回合平均立住步数"); ax2 = ax.twinx(); ax2.plot(t, eps, "--", color="gray", label="探索率 ε")
ax.set_xlabel("训练步数"); ax.set_ylabel("立住步数（满分 500）"); ax2.set_ylabel("ε"); ax.grid(alpha=0.3)
ax.set_title(f"DQN 学习曲线（{tag}）"); fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9)); fig.tight_layout()
fig.savefig(f"out/lesson4_5_{tag}.png", dpi=110)
print(f"已生成 out/lesson4_5_{tag}.png 和 out/dqn_{tag}.pt")
