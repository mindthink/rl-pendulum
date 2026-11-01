"""我们自己的小车倒立摆（cart-pole）仿真。整个课程都用它。

物理模型来自 Barto, Sutton & Anderson (1983)，和 gymnasium 的 CartPole-v1 是同一组方程和参数，
所以可以拿 gymnasium 来核对我们写得对不对（见 lesson1_build_sim.py）。

状态（state）4 个数：
  x          小车位置，m（向右为正）
  x_dot      小车速度，m/s
  theta      杆和竖直方向的夹角，rad（杆顶向右倒为正）
  theta_dot  杆的角速度，rad/s
输入：作用在小车上的水平力 F，N（向右为正）。CartPole-v1 只允许 ±10 N，我们允许 -10~+10 之间任意值，
这样传统控制（PID、LQR）可以输出连续的力。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    gravity: float = 9.8        # 重力加速度，m/s²
    m_cart: float = 1.0         # 小车质量，kg
    m_pole: float = 0.1         # 杆质量，kg
    half_len: float = 0.5       # 杆长的一半，m（杆全长 1.0 m）
    f_max: float = 10.0         # 最大推力，N（电机力气有限）
    dt: float = 0.02            # 时间步，s（50 Hz，和 Microduck 的控制频率一样）
    x_limit: float = 2.4        # 轨道半长，m，出界就算失败
    theta_limit: float = np.radians(12)  # 倾斜超过 12° 算倒了


@dataclass
class CartPole:
    p: Params = field(default_factory=Params)
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def reset(self, state=None, rng=None):
        """不给 state 时，和 gymnasium 一样：4 个数都在 ±0.05 里随机。"""
        if state is None:
            rng = rng or np.random.default_rng()
            state = rng.uniform(-0.05, 0.05, 4)
        self.state = np.array(state, dtype=float)
        return self.state.copy()

    def derivatives(self, s, force):
        """运动方程：给定状态和力，算出加速度。"""
        x, x_dot, th, th_dot = s
        p = self.p
        total_m = p.m_cart + p.m_pole
        sin, cos = np.sin(th), np.cos(th)
        temp = (force + p.m_pole * p.half_len * th_dot ** 2 * sin) / total_m
        th_acc = (p.gravity * sin - cos * temp) / (p.half_len * (4.0 / 3.0 - p.m_pole * cos ** 2 / total_m))
        x_acc = temp - p.m_pole * p.half_len * th_acc * cos / total_m
        return x_acc, th_acc

    def step(self, force):
        """往前走一个时间步（显式欧拉积分，和 gymnasium 一致）。返回新状态、是否失败。"""
        force = float(np.clip(force, -self.p.f_max, self.p.f_max))
        x, x_dot, th, th_dot = self.state
        x_acc, th_acc = self.derivatives(self.state, force)
        dt = self.p.dt
        self.state = np.array([x + dt * x_dot, x_dot + dt * x_acc, th + dt * th_dot, th_dot + dt * th_acc])
        return self.state.copy(), self.failed()

    def failed(self):
        x, _, th, _ = self.state
        return abs(x) > self.p.x_limit or abs(th) > self.p.theta_limit


def simulate(controller, seconds=10.0, state0=None, sim=None, stop_on_fail=True, rng=None):
    """用一个控制器跑一段。controller(state) -> 力 F。返回每一步的 时间、状态、力。"""
    sim = sim or CartPole()
    s = sim.reset(state0, rng)
    ts, states, forces = [0.0], [s], []
    for k in range(int(seconds / sim.p.dt)):
        f = float(np.clip(controller(s), -sim.p.f_max, sim.p.f_max))
        s, fail = sim.step(f)
        ts.append((k + 1) * sim.p.dt); states.append(s); forces.append(f)
        if fail and stop_on_fail:
            break
        if abs(s[2]) > np.pi / 2:   # 杆砸到地面（模型里没有地面，这里手动停）
            break
    forces.append(forces[-1] if forces else 0.0)
    return np.array(ts), np.array(states), np.array(forces)
