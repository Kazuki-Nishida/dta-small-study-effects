"""Normal-approximation (REML) bivariate meta-regression with a size covariate: the starting
value of the likelihood fit.

    y_i = (eta_i, phi_i) | s_i ~ N2(mu + beta (s_i - s_bar), Sigma_res + D_i),

with y_i the continuity-corrected empirical logits and D_i = diag(v_eta,i, v_phi,i) the
within-study variances.  Two variance conventions are available:

    "smoothed"  v_eta,i = 1 / {n1_i pbar_e (1 - pbar_e)}, pbar_e = sum TP_i / sum n1_i
                (each study's own group sizes, the pooled proportion: the weights then
                depend on size only, which removes the mean-variance coupling of the
                count-based variances).  This form starts the likelihood fit of ``glmm``.
    "count"     v_eta,i = 1/(TP_i + 1/2) + 1/(FN_i + 1/2)  (delta method with continuity
                correction; the conventional shortcut).

``fit`` is an unconstrained REML fit over (log sig_eta, log sig_phi, artanh rho) with the
means profiled by GLS; ``analyse`` wraps it for one review or one replicate.
"""
import numpy as np
from scipy.optimize import minimize


def logits(TP, FN, FP, TN):
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    eta = np.log((TP + .5) / (FN + .5)); phi = np.log((FP + .5) / (TN + .5))
    return eta, phi


def variances(TP, FN, FP, TN, kind="smoothed"):
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    if kind == "smoothed":
        pe = TP.sum() / n1.sum(); pf = FP.sum() / n0.sum()
        return 1 / (n1 * pe * (1 - pe)), 1 / (n0 * pf * (1 - pf))
    if kind == "count":
        return 1 / (TP + .5) + 1 / (FN + .5), 1 / (FP + .5) + 1 / (TN + .5)
    raise ValueError(kind)


def ess_s(TP, FN, FP, TN):
    """Effective sample size ESS_i = 4 n1 n0 / (n1 + n0) and s_i = 1/sqrt(ESS_i)."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    ess = 4 * n1 * n0 / (n1 + n0)
    return ess, 1 / np.sqrt(ess)


def _nll_factory(y1, y2, v1, v2, X):
    """Returns nll(par) for par = (log sig_e, log sig_f, artanh rho) with the means profiled
    by GLS; X is the (k, p) per-coordinate design, shared by both coordinates.  Closed-form
    2 x 2 inversions; REML correction -1/2 log det of the mean-model information."""
    k, p = X.shape

    def core(se2, sf2, rho):
        c12 = rho * np.sqrt(se2 * sf2)
        V11 = se2 + v1; V22 = sf2 + v2; V12 = np.full(k, c12)
        det = V11 * V22 - V12 ** 2
        if np.any(det <= 0):
            return None
        i11 = V22 / det; i22 = V11 / det; i12 = -V12 / det
        A = np.zeros((2 * p, 2 * p)); rhs = np.zeros(2 * p)
        G11 = X.T @ (X * i11[:, None]); G22 = X.T @ (X * i22[:, None])
        G12 = X.T @ (X * i12[:, None])
        A[:p, :p] = G11; A[p:, p:] = G22; A[:p, p:] = G12; A[p:, :p] = G12
        rhs[:p] = X.T @ (i11 * y1 + i12 * y2)
        rhs[p:] = X.T @ (i22 * y2 + i12 * y1)
        try:
            beta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None
        m1 = X @ beta[:p]; m2 = X @ beta[p:]
        r1 = y1 - m1; r2 = y2 - m2
        quad = np.sum(i11 * r1 ** 2 + 2 * i12 * r1 * r2 + i22 * r2 ** 2)
        s, ld = np.linalg.slogdet(A)
        if s <= 0:
            return None
        ll = -0.5 * (np.sum(np.log(det)) + quad + ld)
        return ll, beta, A

    def nll(par):
        se2, sf2 = np.exp(2 * par[0]), np.exp(2 * par[1])
        rho = np.tanh(par[2]) if len(par) == 3 else 0.0
        out = core(se2, sf2, rho)
        return 1e10 if out is None else -out[0]

    return nll, core


def _starts(y1, y2, v1, v2):
    se0 = np.sqrt(max(np.var(y1) - np.mean(v1), 0.04))
    sf0 = np.sqrt(max(np.var(y2) - np.mean(v2), 0.04))
    r0 = float(np.clip(np.corrcoef(y1, y2)[0, 1], -0.9, 0.9))
    return [[np.log(se0), np.log(sf0), np.arctanh(r0)], [np.log(0.5), np.log(0.5), 0.0]]


def fit(y1, y2, v1, v2, x):
    """Unconstrained REML fit over (log sig_eta, log sig_phi, artanh rho) from two starts
    (Nelder-Mead).  Returns sig_e, sig_f, rho, beta = (mu_eta, beta_eta, mu_phi, beta_phi),
    cov_beta, k and the REML log-likelihood."""
    k = len(y1)
    X = np.column_stack([np.ones(k), x])
    nll, core = _nll_factory(y1, y2, v1, v2, X)
    best = None
    for x0 in _starts(y1, y2, v1, v2):
        res = minimize(nll, x0, method="Nelder-Mead",
                       options=dict(maxiter=3000, xatol=1e-5, fatol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    par = best.x
    se, sf, rho = float(np.exp(par[0])), float(np.exp(par[1])), float(np.tanh(par[2]))
    ll, beta, A = core(se * se, sf * sf, rho)
    cov_beta = np.linalg.inv(A)
    return dict(sig_e=se, sig_f=sf, rho=rho, beta=beta, cov_beta=cov_beta, k=k, ll=ll)


def analyse(TP, FN, FP, TN, kind="smoothed"):
    """The normal-approximation analysis of one review / one replicate: size-adjusted REML
    fit on the empirical logits.  The entry ``fit`` starts the likelihood fit
    (``glmm.start_from_normal``)."""
    eta, phi = logits(TP, FN, FP, TN)
    v_e, v_f = variances(TP, FN, FP, TN, kind)
    ess, s = ess_s(TP, FN, FP, TN)
    f = fit(eta, phi, v_e, v_f, s - s.mean())
    C = f["cov_beta"]
    return dict(k=f["k"], beta_eta=float(f["beta"][1]), se_eta=float(np.sqrt(C[1, 1])),
                beta_phi=float(f["beta"][3]), se_phi=float(np.sqrt(C[3, 3])),
                rho=f["rho"], sig_e=f["sig_e"], sig_f=f["sig_f"], fit=f)
