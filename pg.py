"""第 5 课：策略梯度（policy gradient），REINFORCE 算法。

策略（policy）是一个小神经网络：输入 4 个状态，输出推力的“平均值” μ；实际推力从正态分布 N(μ, σ²) 里抽。
  - 有随机性才会探索：同一个状态，这次推多一点、下次推少一点，看哪种拿分多
  - σ 也是要学的参数：一开始大（多探索），学会以后自己变小
  - 推力是连续的，不再只有 ±10 N 两种

REINFORCE 的更新规则（一句话）：
  这一步之后拿到的分比平均好，就把“这一步做的动作”的概率调高；比平均差，就调低。
  数学上：参数 += 学习率 × Σ_t ∇log π(a_t | s_t) × A_t
  其中 A_t 是优势（advantage）：从第 t 步起实际拿到的总分（打折扣）减去平均水平。
"""
import numpy as np
import torch
import torch.nn as nn

from env import CartPoleMJ

F_MAX = 10.0


class GaussianPolicy(nn.Module):
    def __init__(self, hidden=32, log_std=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.tensor([log_std]))      # σ = e^1 ≈ 2.7 N 起步
        self.scale = torch.tensor([2.4, 2.0, np.radians(12), 2.0], dtype=torch.float32)  # 把 4 个状态缩放到差不多的大小

    def dist(self, s):
        mu = F_MAX * torch.tanh(self.net(s / self.scale))            # 平均推力，限制在 ±10 N 内
        return torch.distributions.Normal(mu, self.log_std.exp())

    def act(self, s, deterministic=False):
        with torch.no_grad():
            d = self.dist(torch.as_tensor(s, dtype=torch.float32))
            a = d.mean if deterministic else d.sample()
        return float(a.clamp(-F_MAX, F_MAX))


def step_reward(s):
    """和第 3 课的“塑形奖励”一样：活着有分，离中间、离竖直越远扣得越多。"""
    return 1.0 - 0.5 * (s[0] / 2.4) ** 2 - 0.5 * (s[2] / np.radians(12)) ** 2


def rollout(policy, env, rng, max_steps=500):
    s = env.reset(rng=rng); states, actions, rewards = [], [], []
    for _ in range(max_steps):
        a = policy.act(s)
        states.append(s); actions.append(a)
        s, failed = env.step(a)
        rewards.append(0.0 if failed else step_reward(s))
        if failed:
            break
    return np.array(states), np.array(actions), np.array(rewards)


def returns_to_go(r, gamma):
    g, out = 0.0, np.zeros_like(r)
    for t in range(len(r) - 1, -1, -1):
        g = r[t] + gamma * g; out[t] = g
    return out


def train(iters=150, batch=16, gamma=0.99, lr=1e-2, seed=0, log=10, max_steps=500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    policy = GaussianPolicy(); opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = CartPoleMJ(); history = []
    for it in range(1, iters + 1):
        S, A, G, lens, rets = [], [], [], [], []
        for _ in range(batch):                                  # 先用当前策略玩 batch 个回合
            s, a, r = rollout(policy, env, rng, max_steps)
            S.append(s); A.append(a); G.append(returns_to_go(r, gamma)); lens.append(len(r)); rets.append(r.sum())
        S = torch.as_tensor(np.concatenate(S), dtype=torch.float32)
        A = torch.as_tensor(np.concatenate(A), dtype=torch.float32).unsqueeze(1)
        G = np.concatenate(G)
        adv = torch.as_tensor((G - G.mean()) / (G.std() + 1e-8), dtype=torch.float32).unsqueeze(1)   # 减平均、除标准差
        loss = -(policy.dist(S).log_prob(A) * adv).mean()       # 最小化它 = 让“好动作”的概率变大
        opt.zero_grad(); loss.backward(); opt.step()
        history.append((it, float(np.mean(lens)), float(np.mean(rets)), float(policy.log_std.exp())))
        if log and it % log == 0:
            print(f"  第 {it:4d} 轮  平均立住 {np.mean(lens):6.1f} 步  平均总分 {np.mean(rets):6.1f}  σ = {policy.log_std.exp().item():.2f} N", flush=True)
    return policy, history
