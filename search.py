"""第 3 课用的工具：给一组增益打分，以及两种最朴素的“学习”方法。

控制器形式不变：F = K · [x, ẋ, θ, θ̇]（4 个增益）。程序不知道物理，只能试。
打分（reward）：每回合杆没失败就每步 +1，最多 500 步（10 s）。随机几个初始状态取平均。
"""
import numpy as np

from env import CartPoleMJ

MAX_STEPS = 500


def episode_return(K, s0, env, reward="survive", max_steps=MAX_STEPS):
    """跑一回合，返回总分。reward="survive"：每活一步 +1。
    reward="shaped"：每步 1 − 0.5·(x/2.4)² − 0.5·(θ/12°)²，活着、而且离中间和竖直越近分越高。"""
    s = env.reset(s0)
    total = 0.0
    for _ in range(max_steps):
        s, failed = env.step(float(np.dot(K, s)))
        if failed:
            break
        if reward == "survive":
            total += 1.0
        else:
            total += 1.0 - 0.5 * (s[0] / 2.4) ** 2 - 0.5 * (s[2] / np.radians(12)) ** 2
    return total


def score(K, starts, env=None, reward="survive"):
    env = env or CartPoleMJ()
    return float(np.mean([episode_return(K, s0, env, reward) for s0 in starts]))


def make_starts(n=5, seed=0):
    """固定几个随机初始状态（4 个数都在 ±0.05 里），所有候选都用同一组，比较才公平。"""
    return np.random.default_rng(seed).uniform(-0.05, 0.05, (n, 4))


def random_guess(n_trials, low=-60, high=60, seed=0, reward="survive", starts=None, log=None):
    """方法 1：纯瞎猜。每次独立随机抽 4 个增益，留下分最高的。"""
    rng = np.random.default_rng(seed); starts = make_starts() if starts is None else starts
    env = CartPoleMJ(); best_K, best = None, -np.inf; history = []
    for i in range(n_trials):
        K = rng.uniform(low, high, 4)
        sc = score(K, starts, env, reward)
        if sc > best:
            best_K, best = K, sc
        history.append((i, sc, best))
        if log and (i + 1) % log == 0:
            print(f"  试了 {i + 1:4d} 组，目前最好 {best:6.1f} 分", flush=True)
    return best_K, best, history


def hill_climb(n_iters, step=5.0, seed=0, reward="survive", starts=None, K0=None, log=None):
    """方法 2：爬山（hill climbing）。从当前最好的增益出发，随机挪一小步；分数更高就搬过去，否则留在原地。"""
    rng = np.random.default_rng(seed); starts = make_starts() if starts is None else starts
    env = CartPoleMJ()
    K = np.zeros(4) if K0 is None else np.asarray(K0, float)
    best = score(K, starts, env, reward); history = [(0, best, best)]
    for i in range(1, n_iters + 1):
        cand = K + rng.normal(0, step, 4)
        sc = score(cand, starts, env, reward)
        if sc >= best:
            K, best = cand, sc
        history.append((i, sc, best))
        if log and i % log == 0:
            print(f"  第 {i:4d} 步，目前最好 {best:6.1f} 分，增益 {np.round(K, 1)}", flush=True)
    return K, best, history
