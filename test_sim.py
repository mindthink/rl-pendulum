"""仿真底座的自动检查。改了 cartpole.xml 或 env.py 以后都要跑一遍，全部 PASS 才能继续做实验。

运行：.venv/bin/python test_sim.py
"""
import numpy as np

from env import CartPoleMJ, DECIMATION, THETA_LIMIT, X_LIMIT, simulate
from sim import CartPole, Params

M_C, M_P, L_HALF, G = 1.0, 0.1, 0.5, 9.8
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")


env = CartPoleMJ(); m = env.model

# 1. 参数和教科书一致
mass_ok = abs(m.body("cart").mass[0] - M_C) < 1e-9 and abs(m.body("pole").mass[0] - M_P) < 1e-9
I_ok = abs(m.body("pole").inertia[1] - M_P * (2 * L_HALF) ** 2 / 12) < 1e-6 and abs(m.body("pole").ipos[2] - L_HALF) < 1e-9
check("参数", mass_ok and I_ok and abs(m.opt.gravity[2] + G) < 1e-9 and abs(env.dt - 0.02) < 1e-12 and env.f_max == 10,
      f"小车 {m.body('cart').mass[0]} kg，杆 {m.body('pole').mass[0]} kg，质心 {m.body('pole').ipos[2]} m，"
      f"绕质心惯量 {m.body('pole').inertia[1]:.6f}（细杆 {M_P / 12:.6f}），重力 {-m.opt.gravity[2]}，控制周期 {env.dt} s，推力 ±{env.f_max} N")

# 2. 符号约定：杆顶向右偏、不推 → 杆继续向右倒，小车被反作用推向左；向右推 → 小车向右加速，杆向左转
env.reset([0, 0, 0.1, 0]); s, _ = env.step(0.0)
env.reset([0, 0, 0, 0]); s2, _ = env.step(10.0)
check("符号约定", s[3] > 0 and s[1] < 0 and s2[1] > 0 and s2[3] < 0,
      f"θ=0.1 不推：θ̇={s[3]:+.3f}、ẋ={s[1]:+.3f}；静止向右推 10 N：ẋ={s2[1]:+.3f}、θ̇={s2[3]:+.3f}")

# 3. 和手写方程一致（手写方程用 0.05 ms 欧拉积分当“参考答案”；步长再细，差距还会按比例缩小）
rng = np.random.default_rng(0); worst = 0.0
for trial in range(20):
    s0 = rng.uniform(-0.05, 0.05, 4); a_env, b_env = CartPoleMJ(), CartPole(p=Params(dt=0.00005))
    a_env.reset(s0); b_env.reset(s0)
    for k in range(25):
        f = rng.uniform(-10, 10); a, _ = a_env.step(f)
        for _ in range(400):
            b, _ = b_env.step(f)
        worst = max(worst, np.degrees(abs(a[2] - b[2])))
check("和运动方程一致", worst < 0.05, f"20 次 × 0.5 s 随机推，杆角度最大相差 {worst:.4f}°（要求 < 0.05°）")


# 4. 能量守恒：不推、没有摩擦，杆从 10° 倒到 80°，机械能应该不变
def energy(s):
    x, xd, th, thd = s
    vx = xd + L_HALF * np.cos(th) * thd; vz = -L_HALF * np.sin(th) * thd
    return 0.5 * M_C * xd ** 2 + 0.5 * M_P * (vx ** 2 + vz ** 2) + 0.5 * (M_P * (2 * L_HALF) ** 2 / 12) * thd ** 2 \
        + M_P * G * L_HALF * np.cos(th)


env.reset([0, 0, np.radians(10), 0]); e0 = energy(env.state); drift = 0.0
while abs(env.state[2]) < np.radians(80):
    s, _ = env.step(0.0); drift = max(drift, abs(energy(s) - e0))
check("能量守恒", drift / e0 < 1e-4, f"杆从 10° 倒到 80°，机械能最大变化 {drift / e0 * 100:.5f}%（要求 < 0.01%）")

# 5. 水平动量守恒：不推时没有水平外力
px = lambda s: M_C * s[1] + M_P * (s[1] + L_HALF * np.cos(s[2]) * s[3])
env.reset([0, 0.3, np.radians(5), 0.5]); p0 = px(env.state); dp = 0.0
for k in range(30):
    s, _ = env.step(0.0); dp = max(dp, abs(px(s) - p0))
check("水平动量守恒", dp < 1e-6, f"不推 0.6 s，水平动量最大变化 {dp:.2e} kg·m/s")

# 6. 推力保持：一个控制周期（4 个物理步）里推力不变；超过 ±10 N 会被截断
env.reset([0, 0, 0, 0]); env.step(25.0)
check("推力截断和保持", abs(env.data.ctrl[0] - 10.0) < 1e-12 and DECIMATION == 4,
      f"要求推 25 N，实际 {env.data.ctrl[0]} N；每次控制走 {DECIMATION} 个 5 ms 物理步")

# 7. 失败判断：|θ| > 12° 或 |x| > 2.4 m
env.reset([0, 0, np.radians(11.9), 0]); f1 = env.failed()
env.reset([0, 0, np.radians(12.1), 0]); f2 = env.failed()
env.reset([2.39, 0, 0, 0]); f3 = env.failed()
env.reset([-2.41, 0, 0, 0]); f4 = env.failed()
check("失败判断", (not f1) and f2 and (not f3) and f4, f"11.9°→{f1}，12.1°→{f2}，x=2.39→{f3}，x=−2.41→{f4}")

# 8. 物理限位：小车停在 ±2.55 m 挡块，杆停在 ±95°；正常控制碰不到
env.reset([0, 0, 0, 0]); xs = [env.step(10.0)[0][0] for _ in range(300)]
env.reset([0, 0, 0.05, 0]); ths = [abs(env.step(0.0)[0][2]) for _ in range(300)]
check("物理限位", max(xs) < 2.6 and max(ths) < np.radians(98),
      f"一直向右推 6 s：小车最远 {max(xs):.3f} m；杆最多倒到 {np.degrees(max(ths)):.1f}°")

# 9. 可重复：同样的初始状态和推力，结果完全一样
r1 = simulate(lambda s: 3 * s[2], seconds=2, state0=[0, 0, 0.03, 0], stop_on_fail=False)[1]
r2 = simulate(lambda s: 3 * s[2], seconds=2, state0=[0, 0, 0.03, 0], stop_on_fail=False)[1]
check("可重复", np.array_equal(r1, r2), "两次运行逐位相同")

# 10. 线性化：对 MuJoCo 做有限差分量出的 20 ms 离散模型，和教科书线性模型离散化的结果一致（第 2 课 LQR 要用）
from linear import continuous, discretize, fd_linearize
Ac, Bc = continuous()
Ad_ref, Bd_ref = discretize(Ac, Bc, env.dt)
Ad, Bd = fd_linearize(CartPoleMJ())
errA = np.abs(Ad - Ad_ref).max(); errB = np.abs(Bd - Bd_ref).max()
check("线性化模型", errA < 1e-6 and errB < 1e-6,
      f"20 ms 一步的 Ad 最大相差 {errA:.1e}，Bd 最大相差 {errB:.1e}；"
      f"连续时间：θ̈ = {Ac[3, 2]:.3f}·θ {Bc[3, 0]:+.3f}·F，ẍ = {Ac[1, 2]:+.3f}·θ {Bc[1, 0]:+.3f}·F")

print(f"\n{sum(results)}/{len(results)} 项通过")
raise SystemExit(0 if all(results) else 1)
