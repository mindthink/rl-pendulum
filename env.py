"""MuJoCo 版小车倒立摆环境。整个课程（传统控制和强化学习）都用它。

状态（state）4 个数：x 小车位置 m，x_dot 小车速度 m/s，theta 杆角度 rad（杆顶向右倒为正），theta_dot 杆角速度 rad/s。
控制：每 20 ms 给一次水平推力 F（-10 ~ +10 N）。MuJoCo 内部每 5 ms 算一步物理，一次控制走 4 步。
失败：|theta| > 12° 或 |x| > 2.4 m。
"""
import os

import mujoco
import numpy as np

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole.xml")
DECIMATION = 4                    # 每次控制走几步物理
THETA_LIMIT = np.radians(12)
X_LIMIT = 2.4


class CartPoleMJ:
    def __init__(self, xml=XML):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * DECIMATION      # 0.02 s
        self.f_max = float(self.model.actuator_ctrlrange[0, 1])

    @property
    def state(self):
        d = self.data
        return np.array([d.qpos[0], d.qvel[0], d.qpos[1], d.qvel[1]])

    def reset(self, state=None, rng=None):
        """不给 state 时，4 个数都在 ±0.05 里随机（和 gymnasium 的 CartPole 一样）。"""
        if state is None:
            rng = rng or np.random.default_rng()
            state = rng.uniform(-0.05, 0.05, 4)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [state[0], state[2]]
        self.data.qvel[:] = [state[1], state[3]]
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def step(self, force):
        self.data.ctrl[0] = float(np.clip(force, -self.f_max, self.f_max))
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
        return self.state, self.failed()

    def kick(self, dx_dot=0.0, dth_dot=0.0):
        """扰动（disturbance）：瞬间改变小车速度和杆角速度，相当于有人推了一下。"""
        self.data.qvel[0] += dx_dot
        self.data.qvel[1] += dth_dot
        mujoco.mj_forward(self.model, self.data)
        return self.state

    def failed(self):
        x, _, th, _ = self.state
        return abs(x) > X_LIMIT or abs(th) > THETA_LIMIT


def simulate(controller, seconds=10.0, state0=None, env=None, stop_on_fail=True, rng=None, kicks=None):
    """用一个控制器跑一段。controller(state) -> 推力 F。返回 时间、状态、推力 三个数组。
    kicks：{时间(s): (小车速度增量, 杆角速度增量)}，到时间就推一下。"""
    env = env or CartPoleMJ()
    s = env.reset(state0, rng)
    kick_at = {int(round(t / env.dt)): v for t, v in (kicks or {}).items()}
    ts, states, forces = [0.0], [s], []
    for k in range(int(round(seconds / env.dt))):
        if k in kick_at:
            s = env.kick(*kick_at[k])
        f = float(np.clip(controller(s), -env.f_max, env.f_max))
        s, fail = env.step(f)
        ts.append((k + 1) * env.dt); states.append(s); forces.append(f)
        if fail and stop_on_fail:
            break
        if abs(s[2]) > np.pi / 2:   # 杆砸到地面（模型里杆和地面不碰撞，这里手动停）
            break
    forces.append(forces[-1] if forces else 0.0)
    return np.array(ts), np.array(states), np.array(forces)
