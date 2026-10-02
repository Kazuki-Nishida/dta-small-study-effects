#!/usr/bin/env python3
"""Figures 1 and 2 (simulation): rejection rates of the Deeks test, the lnDOR trend of the binomial fit and the
proposed latent accuracy test at the baseline settings k = 30, rho = 0.4.
Figure 1 (figures/fig_sim_null.*), no accuracy trend, so that every rejection of the three accuracy procedures is a
false alarm for the accuracy question:
  (a) shape axis lambda in {1/4, 1/2, 1/sqrt2, 1, sqrt2, 2, 4} (logarithmic) under a threshold trend of 0.4;
  (b) threshold-trend strength 0 to 1.6 at lambda = 1/2.
Figure 2 (figures/fig_sim_power.*), the same accuracy trend (delta = 0.5) on the shape axis, so that every rejection of
the three accuracy procedures is a correct detection:
  (a) accuracy trend alone (rho_s = 0);
  (b) accuracy trend together with the threshold trend of 0.4 of Figure 1(a).
The nominal level 0.10 is marked in every panel; each figure has two panels side by side and a common legend below.
The rates are read from the per-setting summaries results/simulation/<cell>.json (1000 replicates each); the same
settings appear in the supplementary tables, so the figures and the tables share one set of summaries (nothing is
recomputed).  Missing settings are skipped."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "results", "simulation")
INK, INK2, BASE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

LAMS = [(0.25, "0.25", "1/4"), (0.5, "0.5", "1/2"), (2 ** -0.5, "0.71", r"$1/\sqrt{2}$"), (1.0, "1.0", "1"),
        (2 ** 0.5, "1.41", r"$\sqrt{2}$"), (2.0, "2.0", "2"), (4.0, "4.0", "4")]
STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
# (procedure, label, colour, marker, line style): the two lnDOR-axis procedures share a hue (the estimator differs),
# the proposed test is the accuracy axis on the same likelihood as the second
PROCS = [("deeks", "Deeks test", BLUE, "^", "-"),
         ("c1", "Binomial-fit lnDOR test", BLUE, "v", "--"),
         ("hsLt", "Proposed latent-accuracy LR test", ORANGE, "o", "-")]


def rate(cell, proc):
    """Rejection rate of one procedure in one setting (None when the setting is not present)."""
    f = os.path.join(SIM, f"{cell}.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        m = json.load(fh)["methods"].get(proc)
    return float(m["rate"]) if m is not None and np.isfinite(m["rate"]) else None


def series_shape(rho_s, delta, proc):
    xs, ys = [], []
    for lam, tag, _ in LAMS:
        r = rate(f"K30-r0.4-l{tag}-rs{rho_s:.1f}-d{delta:.1f}", proc)
        if r is not None:
            xs.append(lam); ys.append(r)
    return np.array(xs), np.array(ys)


def series_thresh(proc):
    xs, ys = [], []
    for rs in STRENGTHS:
        r = rate(f"K30-r0.4-l0.5-rs{rs:.1f}-d0.0", proc)
        if r is not None:
            xs.append(rs); ys.append(r)
    return np.array(xs), np.array(ys)


def draw(ax, getter, title, xlabel, first):
    for key, lab, col, mk, ls in PROCS:
        x, y = getter(key)
        if len(x):
            ax.plot(x, y, color=col, lw=1.4, ls=ls, marker=mk, ms=5.5, mfc=col if ls == "-" else "white", mec=col if ls != "-" else "white",
                    mew=0.9 if ls != "-" else 0.6, label=lab if first else None, zorder=3)
    ax.axhline(0.10, color=INK2, lw=0.9, ls=":", zorder=1)
    ax.set_yticks([0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0", "0.1", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_ylim(0, 1.0); ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.5, axis="y"); ax.set_axisbelow(True)


def shape_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks([l for l, _, _ in LAMS]); ax.set_xticklabels([lab for _, _, lab in LAMS], fontsize=8.2)
    ax.set_xlim(0.25 / 1.35, 4 * 1.35)


def two_panel(name, panels):
    """one figure: two panels side by side with the common legend below; panels = [(getter, title, xlabel, is_shape)]"""
    fig = plt.figure(figsize=(7.8, 4.5))
    gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.075, right=0.985, top=0.93, bottom=0.235)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, (getter, title, xlabel, is_shape) in zip(axes, panels):
        draw(ax, getter, title, xlabel, first=(ax is axes[0]))
        if is_shape:
            shape_axis(ax)
        else:
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]); ax.set_xlim(-0.06, 1.66)
    axes[0].set_ylabel("rejection rate at nominal level 0.10")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.8, loc="lower center", ncol=3, handlelength=2.6, columnspacing=1.6, bbox_to_anchor=(0.5, 0.01))
    os.makedirs(os.path.join(HERE, "..", "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "figures", f"{name}.pdf")); fig.savefig(os.path.join(HERE, "..", "figures", f"{name}.png"), dpi=180)
    plt.close(fig)


SHAPE_LABEL = "shape of the summary curve, $\\lambda$  (1 = symmetric)"
two_panel("fig_sim_null", [(lambda proc: series_shape(0.4, 0.0, proc), "(a) Shape $\\lambda$ under a threshold trend of 0.4", SHAPE_LABEL, True),
                           (series_thresh, "(b) Strength of the threshold trend at $\\lambda = 1/2$", "threshold-trend strength (residual SDs per SD of $s_i$)", False)])
two_panel("fig_sim_power", [(lambda proc: series_shape(0.0, 0.5, proc), "(a) Shape $\\lambda$ under the accuracy trend alone", SHAPE_LABEL, True),
                            (lambda proc: series_shape(0.4, 0.5, proc), "(b) Shape $\\lambda$ under both trends", SHAPE_LABEL, True)])
n = sum(len(getter("hsLt")[0]) for getter in (lambda p: series_shape(0.4, 0.0, p), lambda p: series_shape(0.0, 0.5, p), lambda p: series_shape(0.4, 0.5, p), series_thresh))
print(f"fig_sim_null and fig_sim_power written; {n} of 30 settings present (three shape axes of 7 and the strength axis of 9)")
