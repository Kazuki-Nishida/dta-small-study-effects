#!/usr/bin/env python3
"""Applications of the paper (Section 4: Table 2 and Figure 3; supplement Section D).

For each review (FIT: the published counts; IPG: the minimum-count reconstruction, and the maximum-count endpoint):
one size-adjusted binomial fit -> the shape lambda with its interval, the latent accuracy and latent threshold trends
with delta-method standard errors and likelihood-ratio tests (chi^2_1 and signed-root t_{k-2}), the lnDOR trend of
the fit (beta_eta - beta_phi, the Wald contrast on the lnDOR axis), the accuracy-coordinate contrast used to draw the
fitted line of Figure 3, the decomposition of the model's lnDOR trend into its threshold and accuracy parts, and the
Deeks test (slope, SE, p).  Writes results/applications.json.
Usage: python analysis/applications.py [FIT IPG]   (keys to recompute; the others are kept from the existing file)."""
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsroc import glmm, fitting
from hsroc.normal import analyse, logits, ess_s
from hsroc.funnel import deeks_test, wls_slope_and_var

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")
REVIEWS = [("FIT", "fit_crc_refpos.csv", None, "Faecal immunochemical tests for colorectal cancer, screening programmes (reference standard: positive)"),
           ("IPG", "ipg_min.csv", "ipg_max.csv", "Impedance plethysmography for deep-vein thrombosis")]
KEEP = ("lam", "lam_se", "lam_ci", "rho", "shape", "betaA_H", "betaA_1", "gamma_alpha", "gamma_theta")


def analyse_review(csv):
    d = pd.read_csv(csv if os.path.isabs(csv) else os.path.join(DATA, csv))
    TP, FN, FP, TN = (d[c].values.astype(float) for c in ("TP", "FN", "FP", "TN"))
    n1 = TP + FN; n0 = FP + TN; k = len(TP)
    ess, s = ess_s(TP, FN, FP, TN); x = s - s.mean()
    a = analyse(TP, FN, FP, TN, "smoothed")
    o = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(a["fit"]))
    th, V = o["theta"], o["V"]
    q0 = fitting.hsroc_quantities(th, V, k, x=x)
    q = {key: q0[key] for key in KEEP}
    n_a = fitting.fit_null(TP, FN, FP, TN, x, th, "alpha")
    n_t = fitting.fit_null(TP, FN, FP, TN, x, th, "theta")
    q["lrt_alpha"] = fitting.lrt(o["nll"], n_a["nll"], k=k)
    q["lrt_theta"] = fitting.lrt(o["nll"], n_t["nll"], k=k)
    eta_, phi_ = logits(TP, FN, FP, TN)
    sl, pd_ = deeks_test(eta_ - phi_, ess)
    q["deeks_slope"] = float(sl); q["p_deeks"] = float(pd_)
    q["deeks_se"] = float(np.sqrt(wls_slope_and_var(eta_ - phi_, ess)[1]))
    be, bf = float(th[1]), float(th[3]); lam = q["lam"]
    ga, gt = q["gamma_alpha"]["est"], q["gamma_theta"]["est"]
    q.update(k=k, mu_eta=float(th[0]), mu_phi=float(th[2]), beta_eta=be, se_eta=float(np.sqrt(V[1, 1])),
             beta_phi=bf, se_phi=float(np.sqrt(V[3, 3])), nll=o["nll"], n_empty=int(np.sum((TP == 0) | (FN == 0) | (FP == 0) | (TN == 0))),
             deeks_weight=lam ** -0.5 * (lam - 1),
             lnDOR_slope_model=be - bf, lnDOR_threshold_part=lam ** -0.5 * (lam - 1) * gt, lnDOR_accuracy_part=lam ** -0.5 * (lam + 1) / 2 * ga)
    return q


def main():
    out_path = os.path.join(RESULTS, "applications.json")
    only = sys.argv[1:]  # optional list of review keys to (re)compute; others are kept from the existing json
    res = json.load(open(out_path)) if (only and os.path.exists(out_path)) else {}
    for key, csv_min, csv_max, label in REVIEWS:
        if only and key not in only:
            continue
        t0 = time.time()
        r = analyse_review(csv_min); r["label"] = label; r["file"] = os.path.basename(csv_min)
        if csv_max:
            r["max_endpoint"] = analyse_review(csv_max); r["max_endpoint"]["file"] = os.path.basename(csv_max)
        res[key] = r
        print(f"{key}: k={r['k']} lam={r['lam']:.2f} [{r['lam_ci'][0]:.2f},{r['lam_ci'][1]:.2f}] "
              f"| gamma_a {r['gamma_alpha']['est']:+.1f} ({r['gamma_alpha']['se']:.1f}) LRT p_t={r['lrt_alpha']['p_t']:.3f} "
              f"| gamma_t {r['gamma_theta']['est']:+.1f} p_t={r['lrt_theta']['p_t']:.3f} "
              f"| lnDOR fit {r['betaA_1']['est']:+.2f} (SE {r['betaA_1']['se']:.2f}) p={r['betaA_1']['p']:.3f} "
              f"| Deeks slope {r['deeks_slope']:+.2f} (SE {r['deeks_se']:.2f}) p={r['p_deeks']:.4f}; model lnDOR slope {r['lnDOR_slope_model']:+.2f} = "
              f"thr {r['lnDOR_threshold_part']:+.2f} + acc {r['lnDOR_accuracy_part']:+.2f} [{time.time()-t0:.0f}s]", flush=True)
        if csv_max:
            m = r["max_endpoint"]
            print(f"     max-count endpoint: lam={m['lam']:.2f} gamma_a {m['gamma_alpha']['est']:+.1f} LRT p_t={m['lrt_alpha']['p_t']:.3f} "
                  f"gamma_t {m['gamma_theta']['est']:+.1f} p_t={m['lrt_theta']['p_t']:.3f} lnDOR fit p={m['betaA_1']['p']:.3f} deeks p={m['p_deeks']:.3f}", flush=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
