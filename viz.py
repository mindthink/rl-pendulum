"""可视化：把一段仿真画成视频（MP4）和曲线图（PNG）。

画面用 MuJoCo 离屏渲染，文字和推力箭头用 Pillow 叠加。
  video([(标题, ts, states, forces), ...], "out.mp4")   多段并排（2 列），一起播放
  plot([(标题, ts, states, forces), ...], "out.png")    杆角度、小车位置、推力随时间变化
"""
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 640, 360
FONT = None
for f in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
    try:
        FONT = ImageFont.truetype(f, 18)
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
    except OSError:
        pass
plt.rcParams["axes.unicode_minus"] = False


class Painter:
    """用 MuJoCo 离屏渲染（offscreen render）画一帧，再用 Pillow 叠加文字和推力箭头。"""

    def __init__(self):
        import mujoco
        from env import XML
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [0, 0, 0.35]; self.cam.distance = 4.0; self.cam.azimuth = 90; self.cam.elevation = -12

    def frame(self, state, force, title, t):
        x, x_dot, th, th_dot = state
        self.data.qpos[:] = [x, th]; self.data.qvel[:] = [x_dot, th_dot]
        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.cam)
        img = Image.fromarray(self.renderer.render()); d = ImageDraw.Draw(img)
        fallen = abs(th) > np.radians(12) or abs(x) > 2.4
        d.rectangle([0, 0, W, 62], fill=(255, 255, 255))
        d.text((10, 6), title, fill="black", font=FONT)
        d.text((10, 32), f"t = {t:4.2f} s   角度 {np.degrees(th):+6.1f}°   位置 {x:+5.2f} m   推力 {force:+5.1f} N",
               fill="black", font=FONT)
        if fallen:
            d.text((W - 150, 6), "失败", fill=(200, 40, 40), font=FONT)
        if abs(force) > 1e-6:   # 推力箭头，画在画面底部，长度和力成正比
            cx, y, L = W / 2, H - 22, force / 10 * 120
            d.line([(cx, y), (cx + L, y)], fill=(40, 110, 220), width=5)
            d.polygon([(cx + L, y - 7), (cx + L, y + 7), (cx + L + 10 * np.sign(L), y)], fill=(40, 110, 220))
        return np.asarray(img)


def video(runs, out, fps=50, hold_s=1.0):
    """runs: [(标题, ts, states, forces), ...]。多段时排成 2 列。"""
    painter = Painter()
    clips = []
    for title, ts, states, forces in runs:
        c = [painter.frame(s, f, title, t) for t, s, f in zip(ts, states, forces)]
        clips.append(c + [c[-1]] * int(hold_s * fps))
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    cols = 1 if len(clips) == 1 else 2; rows = (len(clips) + cols - 1) // cols
    blank = np.full((H, W, 3), 255, np.uint8)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    for i in range(n):
        tiles = [c[i] for c in clips] + [blank] * (rows * cols - len(clips))
        w.append_data(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]))
    w.close()
    return out


def plot(runs, out, title=""):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, ts, states, forces in runs:
        axes[0].plot(ts, np.degrees(states[:, 2]), label=name)
        axes[1].plot(ts, states[:, 0], label=name)
        axes[2].step(ts, forces, where="post", label=name)
    axes[0].axhline(12, color="r", ls="--", lw=0.8); axes[0].axhline(-12, color="r", ls="--", lw=0.8)
    axes[0].set_ylabel("杆角度（°）\n红线 = 失败界限 ±12°")
    axes[0].set_ylim(-95, 95)
    axes[1].axhline(2.4, color="r", ls="--", lw=0.8); axes[1].axhline(-2.4, color="r", ls="--", lw=0.8)
    axes[1].set_ylabel("小车位置（m）")
    axes[2].set_ylabel("推力（N）"); axes[2].set_xlabel("时间（s）")
    for a in axes:
        a.grid(alpha=0.3); a.legend(fontsize=9, loc="upper right")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out
