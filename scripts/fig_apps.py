#!/usr/bin/env python3
"""Figure 3 (applications): the conventional display above the proposed one, one column per
review.  Rows = axis: top, the Deeks funnel (continuity-corrected empirical lnDOR against s = 1/sqrt(ESS)) with the
ESS-weighted regression line of the Deeks test (solid) and the lnDOR trend of the binomial fit, beta_eta - beta_phi
(dashed; the same axis, the same likelihood as the proposed test); bottom, the accuracy coordinate
eta_i - lam (phi_i - mu_phi) with the fitted latent accuracy trend.  Columns = the two reviews (FIT for colorectal
cancer, IPG for DVT), each headed by the review's name, k and fitted shape.  Fitted lines and p values of the binomial
fit from results/applications.json; the Deeks line is WLS on the data."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
INK, INK2, BASE, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
res = json.load(open(os.path.join(HERE, "..", "results", "applications.json")))
CASES = [("Review 1: FIT for colorectal cancer", "FIT", "fit_crc_refpos.csv"), ("Review 2: IPG for DVT", "IPG", "ipg_min.csv")]


def fmt_p(p, nd=3):
    return "<0.0001" if p < 1e-4 else (f"= {p:.4f}" if p < 0.001 else f"= {p:.{nd}f}")


fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.0))
plt.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.10, wspace=0.32, hspace=0.32)
for j, (name, key, csv) in enumerate(CASES):
    d = pd.read_csv(f"{D}/{csv}")
    TP, FN, FP, TN = (d[c].values.astype(float) for c in ("TP", "FN", "FP", "TN"))
    n1 = TP + FN; n0 = FP + TN; ess = 4 * n1 * n0 / (n1 + n0); s = 1 / np.sqrt(ess)
    eta = np.log((TP + .5) / (FN + .5)); phi = np.log((FP + .5) / (TN + .5))
    r = res[key]; lam = r["lam"]; mu_e = r["mu_eta"]; mu_f = r["mu_phi"]
    xs = np.linspace(0, s.max() * 1.05, 50); xbar = s.mean()
    y = eta - phi
    W = np.diag(ess); X = np.column_stack([np.ones_like(s), s])
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    se_b = np.sqrt(np.linalg.inv(X.T @ W @ X)[1, 1] * (((y - X @ b) ** 2 * ess).sum() / (len(y) - 2)))
    assert abs(b[1] - r["deeks_slope"]) < 0.02, (b[1], r["deeks_slope"])
    c1 = r["betaA_1"]  # the lnDOR trend of the binomial fit (same axis as the Deeks test, same likelihood as the proposed test)
    # each panel carries a legend whose line samples identify the procedures: top row, the Deeks test (solid) and the
    # lnDOR trend of the binomial fit (dashed) on the same axis; bottom row, the proposed test (solid)
    panels = [
        (y, b[0] + b[1] * xs, BLUE, "empirical lnDOR",
         f"Deeks: ${b[1]:+.1f}$ (SE {se_b:.1f}), $p {fmt_p(r['p_deeks'])}$",
         (mu_e - mu_f) + c1["est"] * (xs - xbar),
         f"Fitted lnDOR trend: ${c1['est']:+.1f}$ (SE {c1['se']:.1f}),\n$p {fmt_p(c1['p'])}$"),
        (eta - lam * (phi - mu_f), mu_e + r["betaA_H"]["est"] * (xs - xbar), ORANGE,
         "logit sensitivity at the summary FPR",
         f"Proposed: $\\hat\\gamma_\\alpha = {r['gamma_alpha']['est']:+.1f}$ (SE {r['gamma_alpha']['se']:.1f}),\nLR $p {fmt_p(r['lrt_alpha']['p_t'])}$", None, None),
    ]
    for i, (yy, line, col, ylab, note, line2, note2) in enumerate(panels):
        ax = axes[i, j]
        ax.plot(xs, line, color=col, lw=1.6, ls="-", zorder=2, label=note)
        if line2 is not None:
            ax.plot(xs, line2, color=col, lw=1.3, ls=(0, (4, 2.5)), zorder=2, label=note2)
        ax.scatter(s, yy, s=13, color=col, alpha=0.6, edgecolors="white", linewidths=0.4, zorder=4)
        ax.legend(loc="lower left", fontsize=9.2, frameon=True, fancybox=False, edgecolor=BASE, facecolor="white", framealpha=1,
                  handlelength=1.4, handletextpad=0.45, borderpad=0.35, labelspacing=0.3, borderaxespad=0.2).set_zorder(6)
        ax.set_xlim(0, s.max() * 1.1)
        lo, hi = yy.min(), yy.max(); pad = 0.12 * (hi - lo)
        ax.set_ylim(lo - (4.8 if line2 is not None else 2.5) * pad, hi + pad)   # room for the legend below the points
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    # column header: the review, its size and its fitted shape
    pos = axes[0, j].get_position()
    fig.text((pos.x0 + pos.x1) / 2, 0.965, name, ha="center", va="top", fontsize=10.5, color=INK, fontweight="bold")
    fig.text((pos.x0 + pos.x1) / 2, 0.918, f"$k = {r['k']}$ {'entries' if key == 'FIT' else 'cohorts'}; fitted shape $\\hat\\lambda = {lam:.2f}$\n"
             + ("(asymmetric curve)" if abs(lam - 1) > 0.1 else "(nearly symmetric curve)"), ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.3)
# shared horizontal axis label (both columns use the same axis)
fig.text((axes[1, 0].get_position().x0 + axes[1, 1].get_position().x1) / 2, 0.022,
         "$1/\\sqrt{\\mathrm{ESS}}$  (smaller effective sample sizes to the right)", ha="center", va="bottom", fontsize=9.5, color=INK2)
# row labels: the method
for i, (lab, col) in enumerate((("lnDOR axis:\nDeeks funnel", BLUE), ("Accuracy axis:\naccuracy coordinate", ORANGE))):
    pos = axes[i, 0].get_position()
    fig.text(0.018, (pos.y0 + pos.y1) / 2, lab, ha="left", va="center", fontsize=9.5, color=col, fontweight="bold", rotation=90)
os.makedirs(os.path.join(HERE, "..", "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "..", "figures", "fig_apps.pdf")); fig.savefig(os.path.join(HERE, "..", "figures", "fig_apps.png"), dpi=200)
print("fig_apps written")
