#!/usr/bin/env python3
"""Parametric check of the proposed test in the design of the FIT review (supplement Section D.1).

Counts are generated with the observed group sizes of the 23 entries from the accuracy-null constrained fit
(gamma_alpha = 0, with gamma_theta, the shape, the standard deviations and the correlation at their estimates under
the constraint), and every replicate is analysed exactly as the review was: all seven parameters re-estimated in the
full and in the accuracy-constrained model, the threshold trend free in both.  Under this generating point the Deeks
null is false (beta_lnDOR = lambda^{-1/2}(lambda - 1) gamma_theta), so the Deeks rejection rate here is a false-alarm
rate for the accuracy question, not a level.  Reported: rejection rates at 0.10 and 0.05 of the likelihood-ratio test
(t and chi-square references), of the Wald contrasts of the accuracy coordinate and of the lnDOR trend of the fit, and
of the Deeks test, with Monte-Carlo standard errors; medians of lambda-hat and rho-hat; non-convergence flags and
boundary hits of artanh rho-hat; and, for the Deeks slope, the mean of its WLS standard error against the empirical
standard deviation of the slope.  Writes results/fit_level_checks.json and results/fit_level_checks_acc_null.csv
(one row per replicate).  Usage: python analysis/fit_level_checks.py [R]   (R = 2000 replicates by default; about
50 minutes on one core)."""
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.stats import t as tdist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsroc import data, glmm, normal, fitting, funnel

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
SEED = 12
DESIGN = "acc_null"


def analyse(tp, fn, fp, tn, x, ess, k):
    a = normal.analyse(tp, fn, fp, tn, "smoothed"); f = a["fit"]
    start = np.array([f["beta"][0], f["beta"][1], f["beta"][2], f["beta"][3],
                      np.log(f["sig_e"]), np.log(f["sig_f"]), np.arctanh(np.clip(f["rho"], -0.95, 0.95))])
    g = fitting.fit_fast(tp, fn, fp, tn, x, start)
    nul = fitting.fit_null(tp, fn, fp, tn, x, g["theta"], "alpha"); lr = fitting.lrt(g["nll"], nul["nll"], k=k)
    th = g["theta"]; lam = float(np.exp(th[4] - th[5])); rho = float(np.tanh(th[6]))
    ga = lam ** -0.5 * th[1] - lam ** 0.5 * th[3]; gt = 0.5 * (lam ** -0.5 * th[1] + lam ** 0.5 * th[3])
    e, f_ = data.logits(tp, fn, fp, tn); y = e - f_
    slope, var = funnel.wls_slope_and_var(y, ess); p_deeks = float(2 * tdist.sf(abs(slope / np.sqrt(var)), k - 2))
    return dict(p_lrt_t=lr["p_t"], p_lrt_chi2=lr["p"], p_wald_acc=g["p_hs"], p_c1=g["p_c1"], p_deeks=p_deeks,
                deeks_slope=float(slope), deeks_se=float(np.sqrt(var)), lam=lam, rho=rho, gamma_alpha=float(ga), gamma_theta=float(gt),
                converged=bool(g["converged"]), boundary_rho=bool(abs(th[6]) >= 4 - 1e-6), boundary_sig=bool(th[4] <= -4 + 1e-6 or th[5] <= -4 + 1e-6 or th[4] >= 3 - 1e-6 or th[5] >= 3 - 1e-6))


def main(R=2000):
    d = pd.read_csv(os.path.join(ROOT, "data", "fit_crc_refpos.csv"))
    TP, FN, FP, TN = (d[c].values.astype(float) for c in ("TP", "FN", "FP", "TN")); k = len(TP); n1 = TP + FN; n0 = FP + TN
    s, x = data.size_covariate(TP, FN, FP, TN); ess = data.ess(TP, FN, FP, TN)
    a = normal.analyse(TP, FN, FP, TN, "smoothed"); full = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(a["fit"]))
    thF = full["theta"]; lamF = float(np.exp(thF[4] - thF[5])); gtF = 0.5 * (lamF ** -0.5 * thF[1] + lamF ** 0.5 * thF[3])
    nulF = fitting.fit_null(TP, FN, FP, TN, x, thF, "alpha"); th0 = nulF["theta7"]; lam0 = float(np.exp(th0[4] - th0[5]))
    gt0 = 0.5 * (lam0 ** -0.5 * th0[1] + lam0 ** 0.5 * th0[3])
    th = th0.copy()   # the generating point: the accuracy-null constrained fit
    out = dict(k=int(k), full_fit=dict(theta=[float(v) for v in thF], lam=lamF, gamma_theta=float(gtF), nll=full["nll"]),
               constrained_fit=dict(theta7=[float(v) for v in th0], lam=lam0, gamma_theta=float(gt0), nll=nulF["nll"]),
               generating_points={DESIGN: dict(theta=[float(v) for v in th], lam=float(np.exp(th[4] - th[5])), rho=float(np.tanh(th[6])),
                                               gamma_theta=float(0.5 * (np.exp(-(th[4] - th[5]) / 2) * th[1] + np.exp((th[4] - th[5]) / 2) * th[3])),
                                               beta_lnDOR=float(th[1] - th[3]))})
    print("generating point:", json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in out["generating_points"][DESIGN].items() if kk != "theta"}), flush=True)
    mu_e, be, mu_f, bf, lse, lsf, t = th; se_, sf_, rho = np.exp(lse), np.exp(lsf), np.tanh(t)
    C = np.array([[se_ ** 2, rho * se_ * sf_], [rho * se_ * sf_, sf_ ** 2]]); L = np.linalg.cholesky(C)
    rng = np.random.default_rng(SEED); rows = []; t0 = time.time(); failed = 0
    for r in range(R):
        z = rng.normal(size=(k, 2)) @ L.T; e = mu_e + be * x + z[:, 0]; f = mu_f + bf * x + z[:, 1]
        tp = rng.binomial(n1.astype(int), 1 / (1 + np.exp(-e))).astype(float); fp = rng.binomial(n0.astype(int), 1 / (1 + np.exp(-f))).astype(float)
        fn = n1 - tp; tn = n0 - fp
        try:
            row = analyse(tp, fn, fp, tn, x, ess, k)
        except Exception as ex:
            failed += 1; row = dict(error=str(ex)[:60])
        row["rep"] = r + 1; rows.append(row)
        if (r + 1) % 200 == 0:
            print(f"{DESIGN}: {r + 1}/{R} [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "results", f"fit_level_checks_{DESIGN}.csv"), index=False)
    ok = df.dropna(subset=["p_lrt_t"]); n = len(ok)
    rate = lambda col, a: float((ok[col] < a).mean())
    res = dict(R=int(R), analysed=int(n), failed=int(failed), nonconverged=int((ok["converged"] == False).sum()),
               boundary_rho=int(ok["boundary_rho"].sum()), boundary_sig=int(ok["boundary_sig"].sum()),
               median_lam=float(ok["lam"].median()), median_rho=float(ok["rho"].median()), mc_se_at_0p10=float(np.sqrt(0.1 * 0.9 / n)),
               deeks_slope_mean=float(ok["deeks_slope"].mean()), deeks_slope_sd=float(ok["deeks_slope"].std(ddof=1)), deeks_se_mean=float(ok["deeks_se"].mean()),
               deeks_t_sd=float((ok["deeks_slope"] / ok["deeks_se"]).std(ddof=1)))
    for col in ("p_lrt_t", "p_lrt_chi2", "p_wald_acc", "p_c1", "p_deeks"):
        for a in (0.10, 0.05):
            p = rate(col, a); res[f"rej_{col}_{a:.2f}"] = p; res[f"se_{col}_{a:.2f}"] = float(np.sqrt(p * (1 - p) / n))
    res["seconds"] = round(time.time() - t0); out[DESIGN] = res
    json.dump(out, open(os.path.join(ROOT, "results", "fit_level_checks.json"), "w"), indent=1)
    print(f"RESULT {DESIGN}: n={n} failed={failed} nonconv={res['nonconverged']} boundary_rho={res['boundary_rho']} | at 0.10: LRT_t {res['rej_p_lrt_t_0.10']:.3f} "
          f"chi2 {res['rej_p_lrt_chi2_0.10']:.3f} Wald_acc {res['rej_p_wald_acc_0.10']:.3f} c1 {res['rej_p_c1_0.10']:.3f} Deeks {res['rej_p_deeks_0.10']:.3f} "
          f"| Deeks slope sd {res['deeks_slope_sd']:.2f} vs mean WLS se {res['deeks_se_mean']:.2f}; median lam {res['median_lam']:.2f} rho {res['median_rho']:.3f} [{res['seconds']}s]", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
