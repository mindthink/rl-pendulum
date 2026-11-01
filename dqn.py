"""第 4.5 课：DQN（deep Q-network）= 第 4 课的 Q-learning，把 Q 表换成神经网络。

网络：输入 4 个状态（这次连小车位置 x、ẋ 也看），输出每个动作的 Q 值。动作和第 4 课一样：−10 N、+10 N。
更新规则还是贝尔曼方程：目标 = r + γ · max_a' Q_目标网络(s', a')；让 Q(s, a) 往目标靠。

网络版 Q-learning 很容易学崩，DQN 用两个技巧把它稳住：
  1. 经验回放（experience replay）：每一步 (s, a, r, s', 是否失败) 存进一个大缓存，训练时随机抽 64 条来学，
     打乱前后步骤之间的相关性，旧经验也能反复用。
  2. 目标网络（target network）：算“目标”时用一个每 500 步才同步一次的旧网络，避免网络追着自己跑。
train() 的 replay=False / target=False 可以分别去掉这两个技巧，看会怎样。
"""
import numpy as np
import torch
import torch.nn as nn

from env import CartPoleMJ
from pg import step_reward

ACTIONS = np.array([-10.0, 10.0])
SCALE = torch.tensor([2.4, 2.0, np.radians(12), 2.0], dtype=torch.float32)


class QNet(nn.Module):
    def __init__(self, hidden=64, n_actions=len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, s):
        return self.net(s / SCALE)


def greedy(qnet):
    def ctrl(s):
        with torch.no_grad():
            return float(ACTIONS[int(qnet(torch.as_tensor(s, dtype=torch.float32)).argmax())])
    return ctrl


def train(total_steps=60000, buffer_size=50000, batch=64, gamma=0.99, lr=1e-3, target_every=500,
          eps_start=1.0, eps_end=0.05, eps_steps=20000, learn_start=1000, reward="survive",
          replay=True, target=True, seed=0, log=5000, max_steps=500):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q = QNet(); q_tgt = QNet(); q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    if not replay:
        buffer_size = batch                     # 不回放：缓存里只留最近 64 步，学的永远是刚发生的事
    S = np.zeros((buffer_size, 4), np.float32); A = np.zeros(buffer_size, np.int64)
    R = np.zeros(buffer_size, np.float32); S2 = np.zeros((buffer_size, 4), np.float32); D = np.zeros(buffer_size, np.float32)
    n = ptr = 0
    env = CartPoleMJ(); s = env.reset(rng=rng); ep_len = 0; ep_lens = []; history = []
    for t in range(1, total_steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * t / eps_steps)
        if rng.random() < eps:
            a = int(rng.integers(len(ACTIONS)))
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, failed = env.step(ACTIONS[a]); ep_len += 1
        r = 0.0 if failed else (1.0 if reward == "survive" else step_reward(s2))
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(failed)
        ptr = (ptr + 1) % buffer_size; n = min(n + 1, buffer_size)
        s = s2
        if failed or ep_len >= max_steps:
            ep_lens.append(ep_len); s = env.reset(rng=rng); ep_len = 0
        if t >= learn_start and n >= batch:
            j = rng.integers(0, n, batch) if replay else np.arange(n)
            sb, ab, rb, s2b, db = (torch.as_tensor(x[j]) for x in (S, A, R, S2, D))
            with torch.no_grad():
                nxt = (q_tgt if target else q)(s2b).max(1).values
                y = rb + gamma * (1 - db) * nxt
            qsa = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(qsa, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if target and t % target_every == 0:
            q_tgt.load_state_dict(q.state_dict())
        if log and t % log == 0:
            recent = ep_lens[-20:] if ep_lens else [ep_len]
            history.append((t, float(np.mean(recent)), eps))
            print(f"  第 {t:6d} 步  探索率 ε = {eps:.2f}  最近 20 回合平均立住 {np.mean(recent):6.1f} 步", flush=True)
    return q, history
