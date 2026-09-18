#!/usr/bin/env python3
"""Fitted parameters of the two reviews (supplement Section D, the fitted-parameter table): for FIT and the two IPG
reconstruction endpoints, the seven fitted parameters (means, trends with their standard errors, residual standard
deviations and correlation), the latent residual standard deviations sigma_alpha-hat and sigma_theta-hat, whether any
variance parameter sits at a bound of the optimizer, the latent trends with their likelihood-ratio tests, the lnDOR
trend of the fit and the accuracy-coordinate contrast, the decomposition of the model's lnDOR trend, the Deeks test,
and the ranges of the observed sensitivities and false-positive rates.  Writes results/fitted_parameters.json.
Usage: python analysis/fitted_parameters.py"""
import os, sys, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.stats import t as tdist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hsroc import data, glmm, normal, fitting, funnel

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")


def full_analysis(TP, FN, FP, TN):
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN)); k = len(TP)
    s, x = data.size_covariate(TP, FN, FP, TN); ess = data.ess(TP, FN, FP, TN)
    a = normal.analyse(TP, FN, FP, TN, "smoothed"); o = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(a["fit"]))
    th, V = o["theta"], o["V"]; q = fitting.hsroc_quantities(th, V, k, x=x)
    n_a = fitting.fit_null(TP, FN, FP, TN, x, th, "alpha"); n_t = fitting.fit_null(TP, FN, FP, TN, x, th, "theta")
    lr_a = fitting.lrt(o["nll"], n_a["nll"], k=k); lr_t = fitting.lrt(o["nll"], n_t["nll"], k=k)
    e, f = data.logits(TP, FN, FP, TN); slope, var = funnel.wls_slope_and_var(e - f, ess); p_deeks = float(2 * tdist.sf(abs(slope / np.sqrt(var)), k - 2))
    mu_e, be, mu_f, bf, lse, lsf, t = th; lam = q["lam"]; rho = float(np.tanh(t)); se_, sf_ = np.exp(lse), np.exp(lsf)
    sig_alpha = float(np.sqrt(2 * se_ * sf_ * (1 - rho))); sig_theta = float(np.sqrt(se_ * sf_ * (1 + rho) / 2))
    bounds = dict(log_sig_eta=float(lse), log_sig_phi=float(lsf), artanh_rho=float(t),
                  at_bound=bool(lse <= -4 + 1e-6 or lse >= 3 - 1e-6 or lsf <= -4 + 1e-6 or lsf >= 3 - 1e-6 or abs(t) >= 4 - 1e-6))
    gt = q["gamma_theta"]["est"]
    fpr = FP / (FP + TN); sens = TP / (TP + FN)
    return dict(k=k, n_empty=int(((TP == 0) | (FN == 0) | (FP == 0) | (TN == 0)).sum()), mu_eta=float(mu_e), mu_phi=float(mu_f), sig_eta=float(se_), sig_phi=float(sf_), rho=rho,
                beta_eta=float(be), beta_phi=float(bf), se_eta=float(np.sqrt(V[1, 1])), se_phi=float(np.sqrt(V[3, 3])), lam=lam, lam_ci=q["lam_ci"], shape=q["shape"],
                sig_alpha=sig_alpha, sig_theta=sig_theta, bounds=bounds,
                gamma_alpha=q["gamma_alpha"], gamma_theta=q["gamma_theta"], lrt_alpha=lr_a, lrt_theta=lr_t, betaA_1=q["betaA_1"], betaA_H=q["betaA_H"],
                lnDOR_threshold_part=float(lam ** -0.5 * (lam - 1) * gt), lnDOR_accuracy_part=float(lam ** -0.5 * (lam + 1) * q["gamma_alpha"]["est"] / 2),
                deeks_slope=float(slope), deeks_se=float(np.sqrt(var)), p_deeks=p_deeks,
                fpr_min=float(fpr.min()), fpr_max=float(fpr.max()), sens_min=float(sens.min()), sens_max=float(sens.max()), converged=bool(o["converged"]))


def main():
    out = {}
    for key, path in (("FIT", "data/fit_crc_refpos.csv"), ("IPG", "data/ipg_min.csv"), ("IPG_max", "data/ipg_max.csv")):
        d = pd.read_csv(os.path.join(ROOT, path))
        r = full_analysis(d["TP"], d["FN"], d["FP"], d["TN"]); out[key] = r
        print(f"{key}: k {r['k']} empty {r['n_empty']} lam {r['lam']:.3f} {[round(v,2) for v in r['lam_ci']]} rho {r['rho']:.3f} sig_alpha {r['sig_alpha']:.3f} sig_theta {r['sig_theta']:.3f} at_bound {r['bounds']['at_bound']} | "
              f"ga {r['gamma_alpha']['est']:+.2f} (SE {r['gamma_alpha']['se']:.2f}) p {r['lrt_alpha']['p_t']:.3f}; gt {r['gamma_theta']['est']:+.2f} p {r['lrt_theta']['p_t']:.3f}; c1 {r['betaA_1']['est']:+.2f} p {r['betaA_1']['p']:.3f}; "
              f"Deeks {r['deeks_slope']:+.2f} (SE {r['deeks_se']:.2f}) p {r['p_deeks']:.3f} | FPR {100*r['fpr_min']:.1f}-{100*r['fpr_max']:.1f}%")
    json.dump(out, open(os.path.join(ROOT, "results", "fitted_parameters.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
