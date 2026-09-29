"""The HSROC reading of the size-adjusted bivariate binomial meta-regression.

full fit   : glmm.fit (7 parameters; theta = (mu_e, be, mu_f, bf, lse, lsf, t)); fit_fast is the
             lighter version used inside the simulation (L-BFGS-B from the normal-approximation
             start, numerical Hessian, no polish)
shape      : lam = exp(lse - lsf) = sig_eta / sig_phi, the shape of the summary curve
             (1 = symmetric); beta_H = -log lam
latent trends : the size trends of the two logits, be and bf, read through the HSROC
             parametrization  be = lam^{1/2}(g_theta + g_alpha/2),  bf = lam^{-1/2}(g_theta - g_alpha/2):
             g_alpha is the latent accuracy trend, g_theta the latent threshold trend
null fits  : the same likelihood with the trend coefficients constrained
               null "alpha" : g_alpha = 0 (accuracy null, threshold trend free)
               null "theta" : g_theta = 0 (threshold null, accuracy trend free)
LRT        : 2(nll_null - nll_full) ~ chi^2_1, or the signed root referred to t_{k-2}
contrasts  : delta-method Wald tests of be - c bf at c = lam-hat (the accuracy coordinate,
             = lam^{1/2} g_alpha), c = -lam-hat (threshold) and c = 1 (the lnDOR axis)
SAUC       : area under the summary curve  logit S(u) = lam logit u + lam^{1/2} alpha
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, t as tdist
from scipy.special import expit, logit

from . import glmm

BOUNDS6 = [(-30, 30), (-30, 30), (-200, 200), (-4, 3), (-4, 3), (-4, 4)]


def fit_fast(TP, FN, FP, TN, x, start, hessian_step=glmm.HESSIAN_STEP_SIMULATION):
    """The simulation fit: L-BFGS-B from the supplied start (the smoothed-variance normal-
    approximation fit), numerical Hessian (step ``hessian_step``, 1e-4 for the simulation study;
    see glmm.HESSIAN_STEP_*), delta-method Wald tests.  Returns the parameter
    vector theta, the negative log-likelihood, the fitted shape c_h = lam-hat and the p values
    of the Wald contrasts at c = lam-hat (p_hs, the accuracy coordinate) and c = 1 (p_c1, the
    lnDOR axis on the model)."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    const = glmm._binom_const(TP, n1, FP, n0)
    f = lambda th: glmm.negloglik(th, TP, n1, FP, n0, x, const)
    B = glmm.BOUNDS
    start = np.clip(start, [b[0] for b in B], [b[1] for b in B])
    res = minimize(f, start, method="L-BFGS-B", bounds=B, options=dict(maxiter=500, ftol=1e-12, gtol=1e-6))
    th = res.x.copy(); p = 7
    H = glmm._numerical_hessian(f, th, h=hessian_step)
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = np.full((p, p), np.nan)
    mu_e, be, mu_f, bf, lse, lsf, t = th
    se, sf = np.exp(lse), np.exp(lsf)
    c_h = se / sf
    gc = np.zeros(p); gc[4] = c_h; gc[5] = -c_h          # gradient of lam-hat
    k = len(TP)

    def contrast(c, gvec):
        g = np.zeros(p); g[1] = 1.0; g[3] = -c; g = g - bf * gvec
        est = be - c * bf
        se_ = np.sqrt(max(float(g @ V @ g), 1e-12))
        return 2 * tdist.sf(abs(est / se_), k - 2)

    return dict(p_hs=contrast(c_h, gc), p_c1=contrast(1.0, np.zeros(p)), c_h=c_h,
                theta=th, nll=float(res.fun), converged=bool(res.success))


def _to7(par6, which):
    mu_e, mu_f, g, lse, lsf, t = par6
    lam = np.exp(np.clip(lse, -4, 3) - np.clip(lsf, -4, 3))
    if which == "alpha":      # g = g_theta, g_alpha = 0
        be, bf = np.sqrt(lam) * g, g / np.sqrt(lam)
    else:                     # g = g_alpha, g_theta = 0
        be, bf = np.sqrt(lam) * g / 2, -g / (2 * np.sqrt(lam))
    return np.array([mu_e, be, mu_f, bf, lse, lsf, t])


def fit_null(TP, FN, FP, TN, x, theta_full, which="alpha"):
    """Constrained ML fit (6 parameters). Returns dict(nll, theta7, par6, converged)."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    const = glmm._binom_const(TP, n1, FP, n0)
    mu_e, be, mu_f, bf, lse, lsf, t = theta_full
    lam = np.exp(lse - lsf)
    g_theta = 0.5 * (be / np.sqrt(lam) + bf * np.sqrt(lam))
    g_alpha = be / np.sqrt(lam) - bf * np.sqrt(lam)
    g0 = g_theta if which == "alpha" else g_alpha
    start = np.array([mu_e, mu_f, g0, lse, lsf, t])
    f = lambda p6: glmm.negloglik(_to7(p6, which), TP, n1, FP, n0, x, const)
    start = np.clip(start, [b[0] for b in BOUNDS6], [b[1] for b in BOUNDS6])
    res = minimize(f, start, method="L-BFGS-B", bounds=BOUNDS6,
                   options=dict(maxiter=500, ftol=1e-12, gtol=1e-6))
    # second start: zero trend, to guard against a poor local optimum
    start2 = start.copy(); start2[2] = 0.0
    res2 = minimize(f, start2, method="L-BFGS-B", bounds=BOUNDS6,
                    options=dict(maxiter=500, ftol=1e-12, gtol=1e-6))
    best = res if res.fun <= res2.fun else res2
    return dict(nll=float(best.fun), theta7=_to7(best.x, which), par6=best.x, converged=bool(best.success))


def lrt(nll_full, nll_null, k=None):
    """LR statistic with the chi^2_1 p value and, if k is given, the p value from referring the
    signed root of the statistic to t_{k-2} (a small-sample convention parallel to the Wald tests)."""
    lr = max(2.0 * (nll_null - nll_full), 0.0)
    out = dict(stat=float(lr), p=float(chi2.sf(lr, 1)))
    if k is not None:
        out["p_t"] = float(2 * tdist.sf(np.sqrt(lr), k - 2))
    return out


def sauc(lam, alpha, n=400):
    """Area under logit S(u) = lam logit u + lam^{1/2} alpha, u in (0,1), by midpoint rule on u."""
    u = (np.arange(n) + 0.5) / n
    return float(np.mean(expit(lam * logit(u) + np.sqrt(lam) * alpha)))


def qstar(lam, alpha):
    return float(expit(np.sqrt(lam) * alpha / (1.0 + lam)))


def hsroc_quantities(theta, V, k, x=None):
    """Delta-method quantities of the HSROC reading from a full fit (theta, V)."""
    mu_e, be, mu_f, bf, lse, lsf, tr = theta
    lam = float(np.exp(lse - lsf)); rho = float(np.tanh(tr)); p = 7
    tq = tdist.ppf(0.975, k - 2)

    def test(est, g):
        se = float(np.sqrt(max(float(g @ V @ g), 1e-12)))
        return dict(est=float(est), se=se, p=float(2 * tdist.sf(abs(est / se), k - 2)),
                    ci=[float(est - tq * se), float(est + tq * se)])

    dlam = np.zeros(p); dlam[4] = lam; dlam[5] = -lam
    # accuracy coordinate beta_eta - lam beta_phi  (= lam^{1/2} gamma_alpha)
    g = np.zeros(p); g[1] = 1.0; g[3] = -lam; g = g - bf * dlam
    betaA_H = test(be - lam * bf, g)
    # threshold coordinate beta_eta + lam beta_phi (= 2 lam^{1/2} gamma_theta)
    g = np.zeros(p); g[1] = 1.0; g[3] = lam; g = g + bf * dlam
    betaT_H = test(be + lam * bf, g)
    # the lnDOR axis on the model
    g = np.zeros(p); g[1] = 1.0; g[3] = -1.0
    betaA_1 = test(be - bf, g)
    # shape and lambda
    g = np.zeros(p); g[4] = -1.0; g[5] = 1.0
    shape = test(lsf - lse, g)
    lam_ci = [float(np.exp(-shape["ci"][1])), float(np.exp(-shape["ci"][0]))]
    g = np.zeros(p); g[4] = lam; g[5] = -lam
    lam_se = float(np.sqrt(max(float(g @ V @ g), 1e-12)))
    # latent trends
    l_m, l_p = lam ** -0.5, lam ** 0.5
    g = np.zeros(p); g[1] = l_m; g[3] = -l_p
    g = g + (-0.5 * lam ** -1.5 * be - 0.5 * lam ** -0.5 * bf) * dlam
    gamma_alpha = test(l_m * be - l_p * bf, g)
    g = np.zeros(p); g[1] = 0.5 * l_m; g[3] = 0.5 * l_p
    g = g + 0.5 * (-0.5 * lam ** -1.5 * be + 0.5 * lam ** -0.5 * bf) * dlam
    gamma_theta = test(0.5 * (l_m * be + l_p * bf), g)
    # mean accuracy at x = 0 and the summary curve
    Lam = l_m * mu_e - l_p * mu_f
    out = dict(lam=lam, lam_se=lam_se, lam_ci=lam_ci, rho=rho, shape=shape, betaA_H=betaA_H, betaT_H=betaT_H,
               betaA_1=betaA_1, gamma_alpha=gamma_alpha, gamma_theta=gamma_theta, Lambda=float(Lam),
               sauc0=sauc(lam, Lam), qstar0=qstar(lam, Lam))
    if x is not None:
        xs = np.asarray(x, float)
        for name, xv in (("largest", xs.min()), ("smallest", xs.max())):
            a = Lam + gamma_alpha["est"] * xv
            out[f"sauc_{name}"] = sauc(lam, a); out[f"qstar_{name}"] = qstar(lam, a); out[f"x_{name}"] = float(xv)
    return out
