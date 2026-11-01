"""第 6 课：Actor-Critic 和 PPO（proximal policy optimization）。Microduck 的训练用的就是 PPO。

和第 5 课 REINFORCE 比，改了三件事：
  1. 多一个“评委”网络 Critic：输入状态，估计“从这里往后还能拿多少分”，即价值函数 V(s)。
     优势（advantage）不再用“整局实际得分 − 全体平均”，而是用“这一步实际的结果 − Critic 对这里的估计”，信号干净得多。
     具体用 GAE（generalized advantage estimation）把每一步的估计误差按 γ·λ 打折累加。
  2. 同一批数据反复用：每批数据过 5 遍（epoch）、每遍分 4 个小批量（mini-batch），而不是用一次就扔。
  3. 每次只许改一点：新旧策略对同一个动作的概率之比，被裁剪（clip）在 [0.8, 1.2] 里，防止一步改过头。

超参数大多和 Microduck 的 velstand 配置一样（见拆解第二篇“训练过程：PPO 加行为克隆”）：
  γ 0.99，λ 0.95，裁剪 0.2，每批 5 遍 × 4 个小批量，学习率 1e-3（Microduck 还会按 KL 自动调）。
不一样的两处，都是在这个倒立摆上试出来的：
  - 熵系数 0（Microduck 0.01）：这里加熵奖励会让 σ 降不下来
  - Critic 学“价值 ÷ 50”：每步奖励约 1，价值有几十，直接拟合太慢、优势很嘈杂（价值归一化，value normalization）
"""
import numpy as np
import torch
import torch.nn as nn

from env import CartPoleMJ
from pg import F_MAX, GaussianPolicy, step_reward


class Critic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.scale = torch.tensor([2.4, 2.0, np.radians(12), 2.0], dtype=torch.float32)

    def forward(self, s):
        return self.net(s / self.scale).squeeze(-1)


def collect(policy, critic, env, rng, n_steps, state, max_steps=500, kick=0.0):
    """用当前策略收集 n_steps 步数据（跨越多个回合）。kick > 0 时，每步有 1% 的概率给杆一个 ±kick 以内的随机冲击。"""
    s, ep_len = state
    S, A, R, D, T, V = [], [], [], [], [], []      # D：失败结束；T：到 500 步被截断（没失败）
    ep_lens = []
    for _ in range(n_steps):
        if kick and rng.random() < 0.01:
            s = env.kick(0.0, rng.uniform(-kick, kick))
        a = policy.act(s)
        with torch.no_grad():
            V.append(float(critic(torch.as_tensor(s, dtype=torch.float32))))
        S.append(s); A.append(a)
        s, failed = env.step(a); ep_len += 1
        R.append(0.0 if failed else step_reward(s))
        trunc = (not failed) and ep_len >= max_steps
        D.append(failed); T.append(trunc)
        if failed or trunc:
            T[-1] = trunc
            ep_lens.append(ep_len)
            s, ep_len = env.reset(rng=rng), 0
    with torch.no_grad():
        last_v = float(critic(torch.as_tensor(s, dtype=torch.float32)))
    return dict(S=np.array(S), A=np.array(A), R=np.array(R), D=np.array(D), T=np.array(T), V=np.array(V),
                last_v=last_v, ep_lens=ep_lens), (s, ep_len)


def gae(batch, critic, gamma=0.99, lam=0.95):
    """广义优势估计：δ_t = r_t + γ·V(s_{t+1}) − V(s_t)，A_t = δ_t + γλ·A_{t+1}（回合结束处断开）。"""
    R, D, T, V = batch["R"], batch["D"], batch["T"], batch["V"]
    n = len(R); adv = np.zeros(n); next_adv = 0.0
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            next_v = batch["last_v"]
        else:
            next_v = V[t + 1]
        if D[t]:
            next_v, next_adv = 0.0, 0.0                   # 失败：之后没有分
        elif T[t]:                                        # 截断：之后其实还有分，用 Critic 估计补上
            with torch.no_grad():
                next_v = V[t]                             # 近似：截断时用当前状态的估计
            next_adv = 0.0
        delta = R[t] + gamma * next_v - V[t]
        adv[t] = next_adv = delta + gamma * lam * next_adv
    return adv, adv + V


def train(iters=60, n_steps=2048, epochs=5, minibatches=4, clip=0.2, ent_coef=0.0, lr=1e-3,
          gamma=0.99, lam=0.95, seed=0, log=5, kick=0.0, log_std=1.0, v_scale=50.0, critic_lr=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    policy, critic = GaussianPolicy(log_std=log_std), Critic()
    opt = torch.optim.Adam([{"params": policy.parameters(), "lr": lr},
                            {"params": critic.parameters(), "lr": critic_lr or lr}])
    env = CartPoleMJ(); state = (env.reset(rng=rng), 0); history = []; total_eps = 0
    for it in range(1, iters + 1):
        batch, state = collect(policy, critic, env, rng, n_steps, state, kick=kick)
        batch["V"] = batch["V"] * v_scale; batch["last_v"] *= v_scale      # Critic 学的是缩小 v_scale 倍的价值
        adv, ret = gae(batch, critic, gamma, lam)
        ret = ret / v_scale
        S = torch.as_tensor(batch["S"], dtype=torch.float32)
        A = torch.as_tensor(batch["A"], dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            old_logp = policy.dist(S).log_prob(A).squeeze(1)
        ADV = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        RET = torch.as_tensor(ret, dtype=torch.float32)
        idx = np.arange(n_steps); mb = n_steps // minibatches
        for _ in range(epochs):                                   # 同一批数据用 5 遍
            rng.shuffle(idx)
            for k in range(minibatches):                          # 每遍分 4 个小批量
                j = torch.as_tensor(idx[k * mb:(k + 1) * mb])
                d = policy.dist(S[j]); logp = d.log_prob(A[j]).squeeze(1)
                ratio = torch.exp(logp - old_logp[j])             # 新策略 / 旧策略 对这个动作的概率之比
                surr = torch.min(ratio * ADV[j], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[j])
                loss_pi = -surr.mean() - ent_coef * d.entropy().mean()
                loss_v = ((critic(S[j]) - RET[j]) ** 2).mean()
                opt.zero_grad(); (loss_pi + 0.5 * loss_v).backward()
                nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 1.0)
                opt.step()
        lens = batch["ep_lens"]; total_eps += len(lens)
        mean_len = float(np.mean(lens)) if lens else float("nan")
        history.append((it, it * n_steps, total_eps, mean_len, float(policy.log_std.exp())))
        if log and it % log == 0:
            print(f"  第 {it:3d} 轮  累计 {it * n_steps:6d} 步 / {total_eps:4d} 回合  这一轮结束的回合平均 {mean_len:6.1f} 步"
                  f"  σ = {policy.log_std.exp().item():.2f} N", flush=True)
    return policy, critic, history
