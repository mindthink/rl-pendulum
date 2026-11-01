"""画课程用的示意图。运行：.venv/bin/python docs/make_diagrams.py  →  docs/control_loop.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

for f in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED = "#222", "#666"
C_THETA, C_X, C_SUM, C_PLANT = "#f3a33a", "#4a90d9", "#8e8e8e", "#5aa469"

fig, ax = plt.subplots(figsize=(16, 8.2))
ax.set_xlim(0, 17); ax.set_ylim(0.2, 8.6); ax.axis("off")


def box(x0, y0, w, h, text, color, fs=12):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color + "22", ec=color, lw=2))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def junction(x, y, color):
    ax.add_patch(Circle((x, y), 0.28, fc="white", ec=color, lw=2))
    ax.text(x, y, "Σ", ha="center", va="center", fontsize=12, color=color)


def arrow(pts, color=INK, label=None, lpos=None, lw=1.8):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16))
    if label:
        ax.text(*lpos, label, fontsize=11, color=color, ha="center", va="bottom")


# ---- 两个设定值、两个误差点、两个 PD ----
ax.text(0.15, 6.5, "设定值\nθ_ref = 0", fontsize=11, va="center", color=C_THETA)
ax.text(0.15, 3.5, "设定值\nx_ref = 0", fontsize=11, va="center", color=C_X)
junction(2.0, 6.5, C_THETA); junction(2.0, 3.5, C_X)
arrow([(1.25, 6.5), (1.72, 6.5)], C_THETA); arrow([(1.25, 3.5), (1.72, 3.5)], C_X)
box(3.5, 5.95, 3.4, 1.1, "杆角度回路（angle loop）\nPD：Kp_theta·θ + Kd_theta·θ̇", C_THETA, fs=10.5)
box(3.5, 2.95, 3.4, 1.1, "小车位置回路（position loop）\nPD：Kp_x·x + Kd_x·ẋ", C_X, fs=10.5)
arrow([(2.28, 6.5), (3.5, 6.5)], C_THETA, "误差 e_θ", (2.89, 6.58))
arrow([(2.28, 3.5), (3.5, 3.5)], C_X, "误差 e_x", (2.89, 3.58))
ax.text(5.2, 2.55, "符号是“反”的：小车在右边（x > 0）时，这一项先往右推，\n让杆向左歪，再由角度回路把小车带回中间",
        fontsize=9.5, color=C_X, ha="center", va="top")

# ---- 相加、截断、电机、被控对象 ----
junction(8.0, 5.0, C_SUM)
arrow([(6.9, 6.5), (8.0, 6.5), (8.0, 5.28)], C_THETA, "F_θ", (7.45, 6.55))
arrow([(6.9, 3.5), (8.0, 3.5), (8.0, 4.72)], C_X, "F_x", (7.45, 3.55))
box(8.85, 4.45, 1.75, 1.1, "限幅\n(saturation)\n±10 N", C_SUM, fs=11)
box(11.15, 4.45, 1.55, 1.1, "电机\n(actuator)\nmotor", C_SUM, fs=11)
box(13.25, 4.0, 2.6, 2.0, "被控对象 (plant)\n小车 + 杆\nMuJoCo，5 ms 一步", C_PLANT, fs=12)
arrow([(8.28, 5.0), (8.85, 5.0)], INK, "F", (8.56, 5.05))
arrow([(10.6, 5.0), (11.15, 5.0)], INK)
arrow([(12.7, 5.0), (13.25, 5.0)], INK, "推力", (12.97, 5.05))
ax.annotate("扰动 (disturbance)\n手推一下", xy=(14.55, 6.0), xytext=(14.55, 7.35), ha="center", fontsize=10.5, color="#c0392b",
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8, mutation_scale=16))
ax.text(14.55, 3.75, "同一个 F 既推小车、又让杆转动：\n两个回路通过这里耦合（coupled）", fontsize=9.5, color=C_PLANT, ha="center", va="top")

# ---- 反馈 ----
ax.plot([15.85, 16.5], [5.3, 5.3], color=C_THETA, lw=1.8); ax.plot([15.85, 16.5], [4.7, 4.7], color=C_X, lw=1.8)
arrow([(16.5, 5.3), (16.5, 8.1), (2.0, 8.1), (2.0, 6.78)], C_THETA)
arrow([(16.5, 4.7), (16.5, 1.35), (2.0, 1.35), (2.0, 3.22)], C_X)
ax.text(9.2, 8.17, "反馈 (feedback)：θ、θ̇（每 20 ms 测一次）", fontsize=11, color=C_THETA, ha="center", va="bottom")
ax.text(9.2, 1.2, "反馈 (feedback)：x、ẋ（每 20 ms 测一次）", fontsize=11, color=C_X, ha="center", va="top")
ax.text(1.72, 6.95, "−", fontsize=15, color=C_THETA, ha="right"); ax.text(1.72, 2.9, "−", fontsize=15, color=C_X, ha="right")
ax.text(1.62, 6.52, "+", fontsize=12, color=C_THETA, ha="right", va="bottom"); ax.text(1.62, 3.52, "+", fontsize=12, color=C_X, ha="right", va="bottom")

ax.text(8.5, 0.45, "整个控制器合起来：F = Kp_x·x + Kd_x·ẋ + Kp_theta·θ + Kd_theta·θ̇（再截断到 ±10 N）",
        fontsize=12.5, color=INK, ha="center", va="center")
ax.set_title("倒立摆的闭环控制：两个 PD 回路，共用一个电机", fontsize=16, pad=10)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_loop.png")
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print(out)
