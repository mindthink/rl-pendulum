"""第 6.5 课：SAC（soft actor-critic）。连续推力 + 旧数据反复用（off-policy）。

和 PPO 的区别：
  - PPO（on-policy）：只用当前策略刚收集的数据，用完就扔
  - SAC（off-policy）：每一步都存进缓存（和 DQN 一样），训练时从缓存里随机抽，旧数据反复用 → 学会所需的数据少得多
和 DQN 的区别：
  - 动作是连续的推力（−10 ~ +10 N）。演员（actor）网络输出正态分布，经过 tanh 压进 ±10 N
  - Q 网络输入“状态 + 推力”，输出一个 Q 值；不需要在所有动作里找最大
另外三个要点：
  - 两个 Q 网络，取较小的那个算目标（减轻 Q 值高估）
  - 目标网络不是隔一段整个复制，而是每步慢慢跟（τ = 0.005，软更新）
  - “soft”：奖励里额外加上策略的熵（随机程度），鼓励多探索；熵的权重 α 自动调节
"""
import numpy as np
import torch
import torch.nn as nn

from env import CartPoleMJ
from pg import F_MAX, step_reward

SCALE = torch.tensor([2.4, 2.0, np.radians(12), 2.0], dtype=torch.float32)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(4, 2)

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s / SCALE).chunk(2, -1)
        log_std = log_std.clamp(-5, 2); std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # 经过 tanh 以后的对数概率（换元公式）
        logp = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a * F_MAX, logp

    def act(self, s, deterministic=False):
        with torch.no_grad():
            a, _ = self(torch.as_tensor(s, dtype=torch.float32), deterministic)
        return float(a)


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(5, 1)

    def forward(self, s, a):
        return self.net(torch.cat([s / SCALE, a / F_MAX], -1)).squeeze(-1)


def train(total_steps=30000, buffer=100000, batch=256, gamma=0.99, lr=3e-4, tau=0.005, learn_start=1000,
          seed=0, log=2500, max_steps=500, v_scale=50.0, target_entropy=-1.0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    actor, q1, q2 = Actor(), QNet(), QNet()
    q1t, q2t = QNet(), QNet(); q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True)   # 目标熵越高，α 越不容易降到很低，探索保留得越多
    oa = torch.optim.Adam(actor.parameters(), lr); oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr)
    oal = torch.optim.Adam([log_alpha], lr)
    S = np.zeros((buffer, 4), np.float32); A = np.zeros((buffer, 1), np.float32); R = np.zeros(buffer, np.float32)
    S2 = np.zeros((buffer, 4), np.float32); D = np.zeros(buffer, np.float32); n = ptr = 0
    env = CartPoleMJ(); s = env.reset(rng=rng); ep_len = 0; ep_lens = []; history = []
    for t in range(1, total_steps + 1):
        a = rng.uniform(-F_MAX, F_MAX) if t < learn_start else actor.act(s)
        s2, failed = env.step(a); ep_len += 1
        r = 0.0 if failed else step_reward(s2)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(failed)
        ptr = (ptr + 1) % buffer; n = min(n + 1, buffer); s = s2
        if failed or ep_len >= max_steps:
            ep_lens.append(ep_len); s = env.reset(rng=rng); ep_len = 0
        if t >= learn_start:
            j = rng.integers(0, n, batch)
            sb, ab, rb, s2b, db = (torch.as_tensor(x[j]) for x in (S, A, R, S2, D))
            alpha = log_alpha.exp().detach()
            with torch.no_grad():                                       # 目标：r + γ(min Q目标(s', a') − α·log π(a'|s'))
                a2, logp2 = actor(s2b)
                qn = torch.min(q1t(s2b, a2), q2t(s2b, a2)) - alpha * logp2 / v_scale
                y = rb / v_scale + gamma * (1 - db) * qn                 # Q 网络学“价值 ÷ 50”（和第 6 课一样做归一化）
            lq = ((q1(sb, ab) - y) ** 2).mean() + ((q2(sb, ab) - y) ** 2).mean()
            oq.zero_grad(); lq.backward(); oq.step()
            an, logp = actor(sb)                                        # 演员：让 Q 大、同时保持一定随机性
            la = (alpha * logp / v_scale - torch.min(q1(sb, an), q2(sb, an))).mean()
            oa.zero_grad(); la.backward(); oa.step()
            lal = -(log_alpha * (logp.detach() + target_entropy)).mean()  # 自动调 α：熵低于目标就加大 α
            oal.zero_grad(); lal.backward(); oal.step()
            with torch.no_grad():
                for net, tgt in ((q1, q1t), (q2, q2t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
        if log and t % log == 0:
            recent = ep_lens[-10:] if ep_lens else [ep_len]
            history.append((t, float(np.mean(recent)), float(log_alpha.exp())))
            print(f"  第 {t:6d} 步  最近 10 回合平均立住 {np.mean(recent):6.1f} 步  α = {log_alpha.exp().item():.3f}", flush=True)
    return actor, history
