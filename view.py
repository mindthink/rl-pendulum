"""MuJoCo 窗口实时看倒立摆，同一个窗口右侧叠加曲线：杆角度、杆角速度、小车位置、推力（最近 5 秒）。

运行（macOS 上被动查看器必须用 mjpython 启动）：
  /Users/mac/Development/rl-pendulum/.venv/bin/mjpython /Users/mac/Development/rl-pendulum/view.py
  ... view.py --ctrl pd        用控制器自动推
  ... view.py --ctrl tune      调参模式：两个 PD 回路，Control 面板里 4 个增益滑块
                               F = Kp_x·x + Kd_x·ẋ + Kp_theta·θ + Kd_theta·θ̇
                               （P = 比例，看偏了多少；D = 微分，看变化有多快；这里还没有 I 积分项）
  ... view.py --ctrl tune --gains 0,0,20,0     指定初始增益
  ... view.py --ctrl lqr --R 0.1 --Q 1,1,10,1  用 LQR 从 Q、R 算出增益，填进同样的 4 个滑块
  ... view.py --ctrl mpc --R 0.1 [--limits]    模型预测控制；--limits 把失败线也写进约束
  ... --kick 2.0                               每一轮 t = 1 s 时给杆一个 2.0 rad/s 的冲击（任何控制器都能加）
  ... view.py --ctrl qtable [--qfile out/q_table_seed0.npy]   第 4 课学到的 Q 表（只会 ±10 N 两种推法）

操作：
  右侧 Control 面板拖 push 滑块 = 你自己推小车（没选控制器时）
  双击选中杆，Ctrl + 右键拖动 = 用手推杆一下（扰动）
  空格 = 暂停 / 继续      delete（Backspace）= 重置
  调参模式（--ctrl tune）不用按键：松开滑块 0.4 s 后自动从起点重来，失败 1.5 s 后也自动重来
窗口里的文字和曲线标签只能显示英文（MuJoCo 自带字体没有中文）：
  pole angle 杆角度（°）、pole rate 杆角速度（°/s）、cart pos 小车位置（m）、force 推力（N）
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from env import DECIMATION, THETA_LIMIT, X_LIMIT, XML

CONTROLLERS = {
    "none": None,
    # 第 2 课会讲：先放一个 PD 控制器，让你看看“稳住”是什么样子
    "pd": lambda s: 40 * s[2] + 8 * s[3] + 1.0 * s[0] + 2.0 * s[1],
}

ap = argparse.ArgumentParser()
ap.add_argument("--ctrl", default="none", choices=list(CONTROLLERS) + ["tune", "lqr", "mpc", "qtable"])
ap.add_argument("--qfile", default="out/q_table_seed0.npy", help="--ctrl qtable：第 4 课学到的 Q 表")
ap.add_argument("--kick", type=float, default=0.0, help="每一轮 t = 1 s 时给杆一个角速度冲击（rad/s），用来比较抗扰动能力")
ap.add_argument("--limits", action="store_true", help="--ctrl mpc：把失败线（|θ| ≤ 11°、|x| ≤ 2.3 m）也写进约束")
ap.add_argument("--Q", default="1,1,10,1", help="--ctrl lqr：对 x, ẋ, θ, θ̇ 偏离的惩罚")
ap.add_argument("--R", type=float, default=0.1, help="--ctrl lqr：对用力的惩罚")
ap.add_argument("--x0", type=float, default=0.5, help="--ctrl tune 时小车的起点（m），默认中点右边 0.5 m")
ap.add_argument("--gains", default="0,0,0,0", help="--ctrl tune 的初始增益：Kp_x,Kd_x,Kp_theta,Kd_theta")
import sys
argv = sys.argv[1:]
for i, a in enumerate(argv[:-1]):          # 让 --gains -1,-2,40,8 这种以负数开头的写法也能用
    if a == "--gains":
        argv[i:i + 2] = [f"--gains={argv[i + 1]}"]
        break
args = ap.parse_args(argv)

if args.ctrl == "qtable":
    import os
    from qlearn import Discretizer, greedy_policy
    qf = args.qfile if os.path.isabs(args.qfile) else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.qfile)
    CONTROLLERS["qtable"] = greedy_policy(np.load(qf), Discretizer())
    q_label = f"Q-table ({os.path.basename(qf)})"
else:
    q_label = None

if args.ctrl == "mpc":
    from mpc import MPC
    _mpc_kw = dict(q=[float(v) for v in args.Q.split(",")], r=args.R)
    if args.limits:
        _mpc_kw.update(theta_limit=np.radians(11), x_limit=2.3)
    CONTROLLERS["mpc"] = MPC(**_mpc_kw)
    mpc_label = f"MPC R={args.R}" + (" +limits" if args.limits else " (force limit only)")
else:
    mpc_label = None

if args.ctrl == "lqr":
    # LQR：从 Q、R 算出 4 个增益，填进调参模式的滑块里（之后还可以手动改）
    from controllers import lqr_gains
    K = lqr_gains(q=[float(v) for v in args.Q.split(",")], r=args.R)
    print(f"LQR  Q = diag({args.Q})  R = {args.R}  →  Kp_x {K[0]:.2f}  Kd_x {K[1]:.2f}  Kp_theta {K[2]:.2f}  Kd_theta {K[3]:.2f}", flush=True)
    args.gains = ",".join(f"{k:.4f}" for k in K)
    args.ctrl, lqr_label = "tune", f"LQR Q=({args.Q}) R={args.R}"
else:
    lqr_label = None

if args.ctrl == "tune":
    # 调参模式：在 Control 面板里多加 4 个滑块，当作增益旋钮。它们是“不出力”的执行器（gear = 0），只用来读数。
    spec = mujoco.MjSpec.from_file(XML)
    # 两个 PD 回路：杆角度回路（Kp_theta、Kd_theta）和小车位置回路（Kp_x、Kd_x）
    for name, hi in (("Kp_x", 10), ("Kd_x", 20), ("Kp_theta", 100), ("Kd_theta", 30)):
        a = spec.add_actuator(); a.name = name; a.target = "hinge"; a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gear[0] = 0.0; a.ctrlrange = [-hi, hi]; a.ctrllimited = 1   # 允许负数，好试试“方向反了”会怎样
    model = spec.compile()
    gains0 = [float(v) for v in args.gains.split(",")]
    controller = lambda s: float(np.dot(data.ctrl[1:5], s))     # F = Kp_x·x + Kd_x·ẋ + Kp_θ·θ + Kd_θ·θ̇
else:
    controller = CONTROLLERS[args.ctrl]
    model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)
DT = model.opt.timestep * DECIMATION
HIST = int(5.0 / DT)                      # 最近 5 秒

state = {"paused": False, "reset": True}


def on_key(key):
    if key == 32:      # 空格
        state["paused"] = not state["paused"]
    elif key == 259:   # Backspace
        state["reset"] = True


def make_fig(title, names, colors, y_range):
    f = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(f)
    f.title = title
    f.flg_extend = 0
    f.flg_legend = 1
    f.gridsize[:] = [5, 5]
    f.range[0][:] = [-5, 0]               # 横轴：过去 5 秒
    f.range[1][:] = y_range
    f.figurergba[:] = [0.1, 0.1, 0.12, 0.75]
    f.panergba[:] = [0.1, 0.1, 0.12, 0.6]
    for i, (n, c) in enumerate(zip(names, colors)):
        f.linename[i] = n
        f.linergb[i] = c
    return f


figs = [
    make_fig("pole angle (deg)", ["angle", "+12 limit", "-12 limit"], [[1, .6, .2], [.9, .2, .2], [.9, .2, .2]], [-20, 20]),
    make_fig("pole rate (deg/s)", ["rate"], [[1, .85, .3]], [-150, 150]),
    make_fig("cart pos (m)", ["x", "+2.4", "-2.4"], [[.4, .8, 1], [.9, .2, .2], [.9, .2, .2]], [-3, 3]),
    make_fig("force (N)", ["force"], [[.4, 1, .5]], [-11, 11]),
]
hist = np.zeros((HIST, 4))                # 角度°、角速度°/s、位置 m、推力 N
t_axis = np.linspace(-5, 0, HIST)


def fill(fig, line, values):
    n = len(values)
    fig.linepnt[line] = n
    fig.linedata[line][0:2 * n:2] = t_axis[-n:]
    fig.linedata[line][1:2 * n:2] = values


with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=False) as viewer:
    viewer.cam.lookat[:] = [0, 0, -0.35]; viewer.cam.distance = 6.0; viewer.cam.azimuth = 90; viewer.cam.elevation = -8
    t = 0.0
    last_gains, gains_changed_at, failed_at = None, None, None
    crossed_t = None                          # 这一轮第一次越过失败线的时刻（一旦越线，整轮都算失败）
    while viewer.is_running():
        tick = time.time()
        if args.ctrl == "tune":
            # 不用按键：滑块改了（停手 0.4 s 后）就自动从起点重来；失败 1.5 s 后也自动重来
            g = data.ctrl[1:5].copy()
            if last_gains is not None and np.abs(g - last_gains).max() > 1e-9:
                gains_changed_at = tick
            last_gains = g
            if gains_changed_at is not None and tick - gains_changed_at > 0.4:
                state["reset"] = True; gains_changed_at = None
        if args.kick and t > 6.0:             # 冲击测试：每轮 6 秒，自动重来
            state["reset"] = True
        if args.ctrl != "none":
            bad = abs(data.qpos[1]) > THETA_LIMIT or abs(data.qpos[0]) > X_LIMIT
            if bad and failed_at is None:
                failed_at = tick
            if failed_at is not None and tick - failed_at > 1.5:
                state["reset"] = True
            if not bad:
                failed_at = None
        if state["reset"]:
            keep = data.ctrl[1:].copy()                 # 重置时保留你调好的增益
            mujoco.mj_resetData(model, data)
            data.ctrl[1:] = keep
            if args.ctrl == "tune" and t == 0.0 and not keep.any():
                data.ctrl[1:5] = gains0
            data.qpos[1] = np.radians((0.0 if args.x0 else 5.0) if args.ctrl == "tune" else 1.0)
            if args.ctrl == "tune":
                data.qpos[0] = args.x0        # 调参模式：小车不从中点开始，好看出它回不回中间
            if args.kick:                     # 冲击测试：从正中间、杆竖直开始，t = 1 s 推一下
                data.qpos[:] = [0.0, 0.0]
            mujoco.mj_forward(model, data)
            hist[:] = 0; t = 0.0; state["reset"] = False; failed_at = None; crossed_t = None
        if not state["paused"]:
            if args.kick and abs(t - 1.0) < DT / 2:      # 每一轮 t = 1 s 时推杆一下
                data.qvel[1] += args.kick
            s = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
            with viewer.lock():
                if controller is not None:
                    data.ctrl[0] = float(np.clip(controller(s), -10, 10))
                for _ in range(DECIMATION):
                    mujoco.mj_step(model, data)
            t += DT
            s = np.array([data.qpos[0], data.qvel[0], data.qpos[1], data.qvel[1]])
            hist = np.roll(hist, -1, axis=0)
            hist[-1] = [np.degrees(s[2]), np.degrees(s[3]), s[0], data.ctrl[0]]

        fill(figs[0], 0, hist[:, 0]); fill(figs[0], 1, np.full(HIST, 12.0)); fill(figs[0], 2, np.full(HIST, -12.0))
        fill(figs[1], 0, hist[:, 1])
        fill(figs[2], 0, hist[:, 2]); fill(figs[2], 1, np.full(HIST, X_LIMIT)); fill(figs[2], 2, np.full(HIST, -X_LIMIT))
        fill(figs[3], 0, hist[:, 3])

        vp = viewer.viewport
        if vp is not None:   # 曲线横排在窗口底部，占 30% 高度；3D 画面在上面
            w, h = int(vp.width / 4), int(vp.height * 0.30)
            viewer.set_figures([(mujoco.MjrRect(vp.left + i * w, vp.bottom, w, h), f) for i, f in enumerate(figs)])
        fallen = abs(data.qpos[1]) > THETA_LIMIT or abs(data.qpos[0]) > X_LIMIT
        if fallen and crossed_t is None:
            crossed_t = t
        if state["paused"]:
            status = "PAUSED"
        elif crossed_t is not None:
            status = f"FAILED: crossed the line at {crossed_t:.2f} s" + (" (now back inside)" if not fallen else "")
        else:
            status = "ok"
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          f"controller: {lqr_label or mpc_label or q_label or args.ctrl}\nt = {t:5.2f} s\nstatus: {status}"
                          + (f"\nKp_x {data.ctrl[1]:.1f}  Kd_x {data.ctrl[2]:.1f}  Kp_theta {data.ctrl[3]:.1f}  Kd_theta {data.ctrl[4]:.1f}"
                             if args.ctrl == "tune" else ""),
                          f"angle {np.degrees(data.qpos[1]):+6.1f} deg\nx {data.qpos[0]:+5.2f} m\nforce {data.ctrl[0]:+5.1f} N"))
        viewer.sync()
        time.sleep(max(0.0, DT - (time.time() - tick)))   # 按真实时间播放
