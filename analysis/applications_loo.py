#!/usr/bin/env python3
"""Leave-one-out refits of a review (supplement Section D.1, the FIT review): for each omitted entry, the shape, the
latent trends with their likelihood-ratio p values, the lnDOR trend of the fit (Wald) and the Deeks test.
Writes results/applications_<key>_loo.json.  Usage: python analysis/applications_loo.py [FIT]"""
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsroc import data, glmm, fitting
from hsroc.normal import analyse, logits, ess_s
from hsroc.funnel import deeks_test, wls_slope_and_var

HERE = os.path.dirname(os.path.abspath(__file__)); RESULTS = os.path.join(HERE, "..", "results")


def one_fit(TP, FN, FP, TN):
    k = len(TP); ess, s = ess_s(TP, FN, FP, TN); x = s - s.mean()
    a = analyse(TP, FN, FP, TN, "smoothed"); o = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(a["fit"]))
    th, V = o["theta"], o["V"]; q = fitting.hsroc_quantities(th, V, k, x=x)
    n_a = fitting.fit_null(TP, FN, FP, TN, x, th, "alpha"); n_t = fitting.fit_null(TP, FN, FP, TN, x, th, "theta")
    lr_a = fitting.lrt(o["nll"], n_a["nll"], k=k); lr_t = fitting.lrt(o["nll"], n_t["nll"], k=k)
    e, f = logits(TP, FN, FP, TN); sl, pd_ = deeks_test(e - f, ess)
    return dict(k=k, lam=q["lam"], rho=float(np.tanh(th[6])), ga=q["gamma_alpha"]["est"], ga_se=q["gamma_alpha"]["se"], pa=lr_a["p_t"],
                gt=q["gamma_theta"]["est"], gt_se=q["gamma_theta"]["se"], pt=lr_t["p_t"], c1=q["betaA_1"]["est"], c1_se=q["betaA_1"]["se"],
                pc=q["betaA_1"]["p"], deeks=float(sl), deeks_se=float(np.sqrt(wls_slope_and_var(e - f, ess)[1])), pd=float(pd_), converged=bool(o["converged"]))


def main(key="FIT"):
    csv = os.path.join(HERE, "..", "data", data.REVIEWS[key][0]); d = pd.read_csv(csv)
    TP, FN, FP, TN = (d[c].values.astype(float) for c in ("TP", "FN", "FP", "TN")); names = d["study"].astype(str).tolist() if "study" in d else [str(i) for i in range(len(d))]
    N = TP + FN + FP + TN; t0 = time.time(); rows = []
    for i in range(len(TP)):
        keep = np.arange(len(TP)) != i
        r = one_fit(TP[keep], FN[keep], FP[keep], TN[keep]); r["omitted"] = names[i]; r["N_omitted"] = int(N[i]); rows.append(r)
        print(f"-{names[i]:20s} N {int(N[i]):7d}: lam {r['lam']:.2f} rho {r['rho']:.2f} ga {r['ga']:+.2f} p {r['pa']:.3f} gt {r['gt']:+.2f} p {r['pt']:.3f} "
              f"c1 {r['c1']:+.2f} p {r['pc']:.3f} Deeks {r['deeks']:+.2f} p {r['pd']:.3f}", flush=True)
    full = one_fit(TP, FN, FP, TN)
    largest = rows[int(np.argmax(N))]; worst = max(rows, key=lambda r: r["pa"])
    summ = dict(key=key, full=full, fits=rows,
                lam_min=min(r["lam"] for r in rows), lam_max=max(r["lam"] for r in rows),
                ga_min=min(r["ga"] for r in rows), ga_max=max(r["ga"] for r in rows), pa_min=min(r["pa"] for r in rows), pa_max=max(r["pa"] for r in rows),
                gt_min=min(r["gt"] for r in rows), gt_max=max(r["gt"] for r in rows), pt_min=min(r["pt"] for r in rows), pt_max=max(r["pt"] for r in rows),
                pc_min=min(r["pc"] for r in rows), pc_max=max(r["pc"] for r in rows), pd_min=min(r["pd"] for r in rows), pd_max=max(r["pd"] for r in rows),
                n_pa_below_10=int(sum(r["pa"] < .1 for r in rows)), n_pd_below_10=int(sum(r["pd"] < .1 for r in rows)),
                pa_drop_largest=largest["pa"], pd_drop_largest=largest["pd"], dropped_largest=largest["omitted"],
                pa_max_study=worst["omitted"], seconds=round(time.time() - t0))
    out = os.path.join(RESULTS, f"applications_{key.lower()}_loo.json")
    with open(out, "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"written {out}: accuracy test p<0.10 in {summ['n_pa_below_10']}/{len(rows)}, Deeks p<0.10 in {summ['n_pd_below_10']}/{len(rows)} [{summ['seconds']}s]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FIT")
