"""第 3 课 · 第 1 步：不懂物理，靠“试 → 打分 → 留下好的”找增益。

运行：.venv/bin/python lesson3_search.py
  方法 1 瞎猜：每次随机抽 4 个增益（各在 −60 ~ 60 之间），试 300 组
  方法 2 爬山：从全 0 出发，每次随机挪一小步，分更高就搬过去，走 200 步
打分：杆立住的步数（每步 20 ms，满分 500 = 10 s），5 个随机开局取平均。
  --reward shaped   第 2 步：换一种打分，离中间和竖直越近分越高
输出：out/lesson3_curve_<reward>.png（学习曲线：横轴是试了多少次，纵轴是分数）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import argparse  # noqa: E402

import viz  # noqa: E402,F401  （借用里面的中文字体设置）
from search import hill_climb, random_guess  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--reward", default="survive", choices=["survive", "shaped"],
                help="survive：每活一步 +1；shaped：每步 1 − 0.5·(x/2.4)² − 0.5·(θ/12°)²")
ap.add_argument("--hill-steps", type=int, default=200)
args = ap.parse_args()

os.makedirs("out", exist_ok=True)
print("方法 1：瞎猜 300 组")
K1, b1, h1 = random_guess(300, seed=1, log=50, reward=args.reward)
s1 = np.array([h[1] for h in h1])
print(f"  300 组里 490 分以上的有 {(s1 >= 490).sum()} 组；分数中位数 {np.median(s1):.0f}")
print(f"  找到的最好增益 Kp_x, Kd_x, Kp_theta, Kd_theta = {np.round(K1, 1)}，{b1:.0f} 分")
print(f"方法 2：爬山 {args.hill_steps} 步（从全 0 出发）")
K2, b2, h2 = hill_climb(args.hill_steps, seed=1, log=25, reward=args.reward)
first = next((i for i, _, b in h2 if b >= 490), None)
print(f"  第 {first} 步第一次到 490 分以上；最终增益 {np.round(K2, 1)}，{b2:.1f} 分")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
ax[0].scatter([h[0] for h in h1], s1, s=8, alpha=0.5, label="每一次瞎猜的分数")
ax[0].plot([h[0] for h in h1], [h[2] for h in h1], color="C3", lw=2, label="到目前为止最好的")
ax[0].set_title("方法 1：瞎猜"); ax[0].set_xlabel("试了多少组"); ax[0].set_ylabel("分数（立住的步数，满分 500）")
ax[1].scatter([h[0] for h in h2], [h[1] for h in h2], s=8, alpha=0.5, label="每一步候选的分数")
ax[1].plot([h[0] for h in h2], [h[2] for h in h2], color="C3", lw=2, label="当前位置的分数")
ax[1].set_title("方法 2：爬山"); ax[1].set_xlabel("走了多少步")
for a in ax:
    a.grid(alpha=0.3); a.legend(fontsize=9); a.axhline(500, color="g", ls="--", lw=0.8)
fig.suptitle("学习曲线（learning curve）"); fig.tight_layout(); fig.savefig(f"out/lesson3_curve_{args.reward}.png", dpi=110)
print(f"已生成 out/lesson3_curve_{args.reward}.png")
