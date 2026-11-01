"""第 2 课：传统控制。从只看角度的 P 控制开始，一步步加到 LQR。

运行：.venv/bin/python lesson2_classical.py
输出在 out/ 下：
  lesson2a_p.mp4/.png          A. 只看角度（P）：增益太小会倒，太大会越晃越厉害
  lesson2b_pd_theta.mp4/.png   B. 加上角速度（PD）：杆立住了，但小车一路漂向轨道尽头
  lesson2c_full.mp4/.png       C. 把小车也管起来：要让小车回中间，得先往反方向推
  lesson2d_lqr.mp4/.png        D. LQR 和手调 PD 对比：同一个扰动下谁更快、谁更稳
屏幕上打印每一步的数字。
"""
import os

import numpy as np

from controllers import PD_FULL, PD_THETA, lqr_gains, linear
from env import simulate
from viz import plot, video

os.makedirs("out", exist_ok=True)
DEG5 = [0, 0, np.radians(5), 0]


def summary(name, ts, st, fs):
    bad = (np.abs(st[:, 2]) > np.radians(12)) | (np.abs(st[:, 0]) > 2.4)
    tf = ts[np.argmax(bad)] if bad.any() else None
    print(f"  {name:<34} {'%.2f s 时失败' % tf if tf is not None else '%.0f s 没失败' % ts[-1]}   "
          f"杆最大 {np.degrees(np.abs(st[:, 2]).max()):4.1f}°   小车最远 {np.abs(st[:, 0]).max():.2f} m   "
          f"最大推力 {np.abs(fs).max():4.1f} N")


def settle_time(st, ts):
    ok = (np.abs(st[:, 2]) < np.radians(0.5)) & (np.abs(st[:, 0]) < 0.05)
    if not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return ts[i]


def max_kick(K, lo=0.0, hi=10.0):
    """二分查找：t = 1 s 时给杆一个角速度冲击，最大多少还能救回来（rad/s）。"""
    for _ in range(22):
        mid = (lo + hi) / 2
        ts, _, _ = simulate(linear(K), seconds=6, state0=[0, 0, 0, 0], kicks={1.0: (0, mid)})
        lo, hi = (mid, hi) if ts[-1] >= 5.99 else (lo, mid)
    return lo


# ---------------- A. 只看角度：F = K·θ ----------------
print("A. P 控制：F = K·θ，杆从 5° 开始")
runs = []
for K in (8, 12, 20):
    ts, st, fs = simulate(linear([0, 0, K, 0]), seconds=10, state0=DEG5)
    runs.append((f"P：F = {K}·θ", ts, st, fs)); summary(f"K = {K}", ts, st, fs)
video(runs, "out/lesson2a_p.mp4"); plot(runs, "out/lesson2a_p.png", "A. 只看角度（P 控制）")

# ---------------- B. 加角速度：F = Kp·θ + Kd·θ̇ ----------------
print("B. PD 控制（只管杆）：F = 40·θ + 8·θ̇")
ts, st, fs = simulate(linear(PD_THETA), seconds=20, state0=DEG5, stop_on_fail=False)
runs = [("PD：F = 40·θ + 8·θ̇（不管小车）", ts, st, fs)]; summary("PD 只管杆", ts, st, fs)
xt = next((t for t, s in zip(ts, st) if abs(s[0]) > 2.4), None)
print(f"  小车在 {xt:.2f} s 越过 2.4 m 失败线" if xt else "  小车没出界")
video(runs, "out/lesson2b_pd_theta.mp4"); plot(runs, "out/lesson2b_pd_theta.png", "B. PD 只管杆：杆立住了，小车漂走")

# ---------------- C. 全状态反馈：F = 1.0·x + 2.0·ẋ + 40·θ + 8·θ̇ ----------------
print("C. 全状态反馈（PD + 管小车），小车从 x = 1 m 开始，杆竖直")
ts, st, fs = simulate(linear(PD_FULL), seconds=10, state0=[1.0, 0, 0, 0])
runs = [("全状态 PD：从 x = 1 m 回到中间", ts, st, fs)]; summary("全状态 PD", ts, st, fs)
first = st[1:15, 0] - 1.0
print(f"  前 0.3 s 小车先向右走了 {first.max() * 100:.1f} cm（目标在左边），然后才往左回；"
      f"最远冲过中点到 {st[:, 0].min():+.2f} m")
video(runs, "out/lesson2c_full.mp4"); plot(runs, "out/lesson2c_full.png", "C. 全状态反馈：先往反方向推，再回中间")

# ---------------- D. LQR ----------------
print("D. LQR：从线性模型算出最优增益")
cands = [("手调 PD", PD_FULL),
         ("LQR 温和（R = 1）", lqr_gains(r=1.0)),
         ("LQR 标准（R = 0.1）", lqr_gains(r=0.1)),
         ("LQR 激进（R = 0.01）", lqr_gains(r=0.01))]
print(f"  {'控制器':<18}{'增益 [x, ẋ, θ, θ̇]':<30}{'5° 回稳时间':<12}{'最大推力':<10}{'能救回的最大冲击'}")
runs = []
for name, K in cands:
    ts, st, fs = simulate(linear(K), seconds=10, state0=DEG5)
    stt = settle_time(st, ts)
    print(f"  {name:<18}{str(np.round(K, 1)):<30}{('%.2f s' % stt) if stt else '没回稳':<12}{np.abs(fs).max():<10.1f}{max_kick(K):.2f} rad/s")
    ts, st, fs = simulate(linear(K), seconds=5, state0=[0, 0, 0, 0], kicks={1.0: (0, 1.6)})
    runs.append((f"{name}，t = 1 s 时杆被推 1.6 rad/s", ts, st, fs))
video(runs, "out/lesson2d_lqr.mp4"); plot(runs, "out/lesson2d_lqr.png", "D. 同一个冲击（1.6 rad/s）：手调 PD vs 三种 LQR")
print("已生成 out/lesson2a~d 的视频和曲线")
