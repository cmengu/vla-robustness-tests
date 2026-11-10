# Schematic diagrams for the README: what each stress test does (fig0) and Test 4's design (fig4).
# Run: python3 -I figures/make_diagrams.py
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
BLUE, ORANGE, AQUA, GRAY, INK, MUTED, PALE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890", "#2b2a27", "#6f6e68", "#eceae4"
plt.rcParams.update({"font.size": 9.5, "text.color": INK, "figure.facecolor": "white"})

def box(ax, x, y, w, h, text, fc="white", ec=INK, lw=1.2, fs=9.5, color=INK, r=0.08, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)

def arrow(ax, p, q, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))

def panel(ax, title, xlim, ylim):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=6)

# ------------------------------------------------------------------ fig0: the three stress tests
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.6), dpi=160)

# Test 1: delayed action execution
panel(a1, "Test 1: actions arrive late", (0, 10), (0, 8))
a1.text(0.3, 7.55, "Each action is applied d control steps (0.1 s each)\nafter the camera frame it was computed from.", fontsize=8.5, va="top", color=MUTED)
a1.text(0.5, 6.05, "camera frames", fontsize=8, color=MUTED)
for i in range(5):
    x = 0.5 + i * 1.9
    box(a1, x, 5.0, 1.4, 0.9, f"t+{i}" if i else "t", fc=PALE, ec=GRAY, fs=8.5)
box(a1, 0.5, 2.6, 1.4, 0.9, "policy\nsees t", fc="white", ec=BLUE, fs=8, color=BLUE)
arrow(a1, (1.2, 5.0), (1.2, 3.5), color=BLUE)
box(a1, 8.1, 2.6, 1.4, 0.9, "action\napplied", fc=BLUE, ec=BLUE, fs=8, color="white")
arrow(a1, (8.8, 5.0), (8.8, 3.5), color=BLUE)
arrow(a1, (1.9, 3.05), (8.1, 3.05), color=BLUE)
a1.text(5.0, 3.35, "delay d = 0, 2, 5, 10 or 20 frames", ha="center", fontsize=8.5, color=BLUE)
a1.text(5.0, 2.2, "the scene has moved on by d frames", ha="center", fontsize=8, color=MUTED)
a1.text(0.3, 0.6, "Measured: success at each d, 100 episodes per point.", fontsize=8.5, color=INK)

# Test 2: the target object moves
panel(a2, "Test 2: the target object moves", (0, 10), (0, 8))
a2.text(0.3, 7.55, "Top view. The object is slid 2, 5 or 10 cm sideways,\nbefore the first action or once the gripper is near.", fontsize=8.5, va="top", color=MUTED)
a2.add_patch(Rectangle((0.5, 1.0), 9.0, 5.3, fc=PALE, ec=GRAY, lw=1))
a2.text(0.7, 5.9, "table", fontsize=8, color=MUTED)
box(a2, 0.9, 1.3, 1.4, 0.9, "gripper\nstart", fc="white", ec=INK, fs=8)
ox, oy = 6.0, 3.7
a2.add_patch(Circle((ox, oy), 1.45, fc="none", ec=AQUA, lw=1.3, ls=":"))
a2.add_patch(Circle((ox, oy), 0.33, fc="white", ec=GRAY, lw=1.2, ls="--"))
a2.add_patch(Circle((ox + 1.5, oy), 0.33, fc=ORANGE, ec=ORANGE))
arrow(a2, (ox + 0.4, oy), (ox + 1.1, oy), color=ORANGE, lw=1.6)
a2.text(ox + 0.75, oy - 0.5, "2 / 5 / 10 cm", ha="center", va="top", fontsize=8, color=ORANGE)
a2.text(ox, oy + 0.5, "where the policy\nexpects it", ha="center", va="bottom", fontsize=7.5, color=MUTED)
a2.text(ox, 1.25, "15 cm ring: the \"mid-reach\" shift fires\nwhen the gripper crosses it", ha="center", va="bottom", fontsize=7.5, color=AQUA)
arrow(a2, (2.3, 2.0), (ox - 1.5, oy - 0.4), color=INK, lw=1.1, ls="--")
a2.text(3.4, 2.15, "reach", fontsize=8, color=MUTED, rotation=24)
a2.text(0.3, 0.4, "Measured: success per cell, 20 episodes.", fontsize=8.5, color=INK)

# Test 3: LIBERO-Plus factors
panel(a3, "Test 3: one scene, seven kinds of change", (0, 10), (0, 8))
a3.text(0.3, 7.55, "LIBERO-Plus changes one factor per case: 100 cases per\nfactor, 700 in all, the same cases for both models.", fontsize=8.5, va="top", color=MUTED)
box(a3, 0.4, 3.0, 1.9, 1.5, "LIBERO\ntask", fc=PALE, ec=GRAY, fs=9)
facs = [("camera viewpoint", BLUE), ("robot start pose", BLUE), ("table layout", BLUE),
        ("extra objects", GRAY), ("sensor noise", GRAY), ("language rewording", GRAY), ("lighting", GRAY)]
for i, (f, c) in enumerate(facs):
    y = 6.0 - i * 0.78
    box(a3, 3.3, y, 3.25, 0.6, f, fc="white", ec=c, fs=8.5, color=c if c == BLUE else INK, lw=1.3)
    arrow(a3, (2.3, 3.75), (3.3, y + 0.3), color=c, lw=0.9)
a3.text(6.75, 5.5, "geometry:\nwhere things\nare in the image", fontsize=7.5, color=BLUE, va="center")
a3.text(6.75, 2.3, "appearance\nand wording", fontsize=7.5, color=MUTED, va="center")
a3.text(0.3, 0.5, "Measured: paired success; McNemar test on discordant pairs.", fontsize=8.5, color=INK)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT + "fig0_tests.png"); plt.close(fig)

# ------------------------------------------------------------------ fig4: Test 4 design
fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
panel(ax, "Test 4: the same policy, fine-tuned with and without a future-prediction head", (0, 22), (0, 10.4))
# inputs
box(ax, 0.4, 6.2, 2.6, 1.2, "main camera\nimage (t)", fc=PALE, ec=GRAY, fs=9)
box(ax, 0.4, 4.6, 2.6, 1.2, "wrist image,\ninstruction", fc=PALE, ec=GRAY, fs=9)
# backbone
box(ax, 4.2, 3.6, 4.4, 4.6, "", fc="white", ec=BLUE, lw=1.5, r=0.15)
ax.text(6.4, 7.75, "Qwen3-VL 4B backbone", ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
for i, lab in enumerate(["layers 1 … 34", "layer 35", "layer 36"]):
    y = 6.5 - i * 1.05
    box(ax, 4.6, y, 3.6, 0.75, lab, fc=PALE if i != 1 else "#dbe8fa", ec=GRAY if i != 1 else BLUE, fs=8.5)
arrow(ax, (3.0, 6.8), (4.2, 6.8), color=INK); arrow(ax, (3.0, 5.2), (4.2, 5.2), color=INK)
ax.text(6.4, 3.75, "keys and values of every layer\nare read by the action expert", ha="center", fontsize=7.5, color=MUTED)
# action expert
box(ax, 10.2, 5.6, 3.4, 1.6, "flow-matching\naction expert", fc="white", ec=BLUE, lw=1.5, fs=9.5, color=BLUE)
arrow(ax, (8.6, 6.4), (10.2, 6.4), color=BLUE)
box(ax, 15.0, 5.75, 2.6, 1.3, "actions", fc=BLUE, ec=BLUE, fs=9.5, color="white")
arrow(ax, (13.6, 6.4), (15.0, 6.4), color=BLUE)
box(ax, 18.3, 5.75, 3.3, 1.3, "flow loss\n(all arms)", fc="white", ec=INK, fs=9)
arrow(ax, (17.6, 6.4), (18.3, 6.4), color=INK)
# head branch
ax.text(8.75, 5.1, "output of layer 35 = input of layer 36:\nthe tensor the expert's last K/V come from", fontsize=7.5, color=MUTED, va="top")
arrow(ax, (8.2, 5.45), (10.0, 3.0), color=ORANGE, lw=1.5)
box(ax, 10.0, 1.95, 4.2, 1.95, "head (B and C only)\nmean of 64 main-camera tokens\n→ MLP → 1,024-d vector\ndiscarded after training", fc="#fde6dc", ec=ORANGE, lw=1.5, fs=7.6)
box(ax, 15.2, 2.2, 2.4, 1.4, "prediction", fc=ORANGE, ec=ORANGE, fs=9.5, color="white")
arrow(ax, (14.2, 2.9), (15.2, 2.9), color=ORANGE)
box(ax, 18.3, 2.2, 3.3, 1.4, "+ 0.1 × head loss\n(B: future, C: current)", fc="white", ec=ORANGE, fs=8.2)
arrow(ax, (17.6, 2.9), (18.3, 2.9), color=ORANGE)
# target branch
box(ax, 0.4, 0.5, 2.6, 1.2, "main camera\nimage (t + 1 s)", fc=PALE, ec=GRAY, fs=9)
box(ax, 4.2, 0.35, 4.4, 1.5, "V-JEPA 2 (frozen)\none 1,024-d vector per frame", fc=PALE, ec=GRAY, fs=8.5)
arrow(ax, (3.0, 1.1), (4.2, 1.1), color=INK)
arrow(ax, (1.7, 6.2), (1.7, 1.7), color=GRAY, lw=1.0, ls=":")
ax.text(1.95, 3.9, "same frame\nalso embedded", fontsize=7.2, color=MUTED)
box(ax, 10.0, 0.3, 4.2, 1.3, "target   B: emb(t + 1 s) − emb(t)\n             C: emb(t)", fc="white", ec=GRAY, fs=7.8)
arrow(ax, (8.6, 1.1), (10.0, 0.95), color=GRAY)
arrow(ax, (14.2, 0.95), (19.9, 2.15), color=GRAY, lw=1.0)
ax.text(15.3, 0.9, "standardised; frames with\nno t + 1 s are masked", fontsize=7.2, color=MUTED, va="center")
# arms legend
ax.text(0.4, 9.6, "Arm A: flow loss only.   Arm B: + head predicting the change 1 s ahead.   Arm C: + the same head predicting the current frame (placebo).",
        fontsize=9, color=INK)
ax.text(0.4, 9.0, "All arms: same checkpoint, same 6,000 steps, batch 16; the saved policy is a stock Alpha-S checkpoint and is scored by the same code as Tests 2 and 3.",
        fontsize=8.5, color=MUTED)
fig.tight_layout(); fig.savefig(OUT + "fig4_test4_design.png"); plt.close(fig)
print("ok")
