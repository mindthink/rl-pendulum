"""第 4 课：表格 Q-learning（tabular Q-learning）。

把连续的状态切成格子（离散化，discretization），每个格子、每个动作记一个 Q 值：
  Q[格子, 动作] ≈ 在这个格子里做这个动作，以后总共能拿多少分（打折扣后）
动作只有两个：向左推 10 N、向右推 10 N（和经典 CartPole 一样）。
奖励：每活一步 +1；失败那一步之后没有分。

每走一步，用贝尔曼方程（Bellman equation）更新一个格子：
  目标 = r + γ · max_a' Q[下一个格子, a']        （失败时目标 = r）
  Q[格子, a] ← Q[格子, a] + α · (目标 − Q[格子, a])
选动作用 ε-贪心（ε-greedy）：概率 ε 随便选（探索，exploration），否则选 Q 最大的（利用，exploitation）。
"""
import numpy as np

from env import CartPoleMJ

ACTIONS = np.array([-10.0, 10.0])
# 每个状态量切成几格、范围多大（超出范围的归到最边上的格子）。
# 默认只看杆：x、ẋ 各 1 格（等于不看），θ 6 格（±12°），θ̇ 6 格（±0.87 rad/s ≈ ±50°/s），一共 36 个格子。
# 格子越多，越精细，但要试的次数越多：试过“x、ẋ 各 3 格”的版本，3000 回合内学得明显更差。
BINS = {"x": (1, 2.4), "x_dot": (1, 2.0), "theta": (6, np.radians(12)), "theta_dot": (6, 0.87)}


class Discretizer:
    def __init__(self, bins=BINS):
        self.n = [b[0] for b in bins.values()]
        self.lim = np.array([b[1] for b in bins.values()])
        self.edges = [np.linspace(-l, l, n + 1)[1:-1] for n, l in zip(self.n, self.lim)]

    def __call__(self, s):
        return tuple(int(np.digitize(v, e)) for v, e in zip(s, self.edges))

    @property
    def shape(self):
        return tuple(self.n)


def train(episodes=3000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.02, seed=0, max_steps=500, log=0):
    rng = np.random.default_rng(seed)
    disc = Discretizer(); env = CartPoleMJ()
    Q = np.zeros(disc.shape + (len(ACTIONS),))
    lengths = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start * (1 - ep / (0.6 * episodes)))      # 前 60% 的回合里，探索从 100% 降到 2%
        s = env.reset(rng=rng); c = disc(s); steps = 0
        for _ in range(max_steps):
            a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(np.argmax(Q[c]))
            s, failed = env.step(ACTIONS[a]); steps += 1
            c2 = disc(s)
            target = 1.0 if failed else 1.0 + gamma * Q[c2].max()
            Q[c + (a,)] += alpha * (target - Q[c + (a,)])
            c = c2
            if failed:
                break
        lengths.append(steps)
        if log and (ep + 1) % log == 0:
            print(f"  第 {ep + 1:5d} 回合  探索率 ε = {eps:.2f}  最近 {log} 回合平均立住 {np.mean(lengths[-log:]):6.1f} 步", flush=True)
    return Q, disc, np.array(lengths)


def greedy_policy(Q, disc):
    """训练完以后只用 Q 最大的动作，不再探索。返回一个控制器 F = policy(s)。"""
    return lambda s: float(ACTIONS[int(np.argmax(Q[disc(s)]))])
