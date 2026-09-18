"""Size-adjusted bivariate binomial meta-regression: the likelihood and the full fit.

Model (Chu and Cole, 2006, with a study-size covariate):

    TP_i ~ Bin(n1_i, expit(eta_i)),   FP_i ~ Bin(n0_i, expit(phi_i)),
    (eta_i, phi_i) ~ N2(mu + beta x_i, Sigma),   x_i = s_i - s_bar,   s_i = 1/sqrt(ESS_i).

Parameter vector theta = (mu_eta, beta_eta, mu_phi, beta_phi, log sig_eta, log sig_phi,
artanh rho), seven parameters.  The marginal likelihood of each study is a two-dimensional
integral, evaluated by adaptive Gauss-Hermite quadrature: the mode of the (log-concave)
integrand is found by damped Newton iteration, the integrand is re-centred at the mode and
rescaled by the Cholesky factor of the inverse negative Hessian there, and a Q x Q product
rule is applied.

The HSROC reading of the fitted trends (shape lambda = sig_eta / sig_phi, latent accuracy and
latent threshold trends, constrained fits and likelihood-ratio tests) is in ``fitting``.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from scipy.special import gammaln, expit

Q_NODES = 9
_x, _w = hermegauss(Q_NODES)
_w = _w / _w.sum()
_ZX, _ZY = np.meshgrid(_x, _x, indexing="ij")
Z = np.column_stack([_ZX.ravel(), _ZY.ravel()])          # (Q^2, 2)
W = np.outer(_w, _w).ravel()                             # (Q^2,)
Z2H = 0.5 * np.sum(Z ** 2, axis=1)                       # |z|^2 / 2

BOUNDS = [(-30, 30), (-200, 200), (-30, 30), (-200, 200), (-4, 3), (-4, 3), (-4, 4)]


def _binom_const(TP, n1, FP, n0):
    return (gammaln(n1 + 1) - gammaln(TP + 1) - gammaln(n1 - TP + 1)
            + gammaln(n0 + 1) - gammaln(FP + 1) - gammaln(n0 - FP + 1))


def _logint(u, m, Si, TP, n1, FP, n0):
    """Log integrand (up to constants) at u, per study."""
    d = u - m
    quad = Si[0, 0] * d[:, 0] ** 2 + 2 * Si[0, 1] * d[:, 0] * d[:, 1] + Si[1, 1] * d[:, 1] ** 2
    return (TP * u[:, 0] - n1 * np.logaddexp(0, u[:, 0])
            + FP * u[:, 1] - n0 * np.logaddexp(0, u[:, 1]) - 0.5 * quad)


def negloglik(theta, TP, n1, FP, n0, x, const, newton_iter=40):
    """Negative marginal log-likelihood by adaptive Gauss-Hermite quadrature."""
    k = len(TP)
    mu_e, be, mu_f, bf, lse, lsf, t = theta
    lse = np.clip(lse, -4, 3); lsf = np.clip(lsf, -4, 3); t = np.clip(t, -4, 4)
    se, sf, rho = np.exp(lse), np.exp(lsf), np.tanh(t)
    det = max(se * se * sf * sf * (1 - rho * rho), 1e-12)
    Si = np.array([[sf * sf, -rho * se * sf], [-rho * se * sf, se * se]]) / det   # Sigma^{-1}
    m = np.column_stack([mu_e + be * x, mu_f + bf * x])                       # (k, 2)
    # damped Newton for the per-study mode of the log-concave integrand
    u = m.copy()
    fu = _logint(u, m, Si, TP, n1, FP, n0)
    for _ in range(newton_iter):
        pe = expit(u[:, 0]); pf = expit(u[:, 1])
        d = u - m
        g = np.column_stack([TP - n1 * pe, FP - n0 * pf]) - d @ Si
        h11 = -n1 * pe * (1 - pe) - Si[0, 0]; h22 = -n0 * pf * (1 - pf) - Si[1, 1]; h12 = -Si[0, 1]
        dh = h11 * h22 - h12 * h12
        step = -np.column_stack([(h22 * g[:, 0] - h12 * g[:, 1]) / dh,
                                 (-h12 * g[:, 0] + h11 * g[:, 1]) / dh])
        lam = np.ones(k)
        for _h in range(8):                       # step halving where the integrand does not increase
            un = u + lam[:, None] * step
            fn = _logint(un, m, Si, TP, n1, FP, n0)
            bad = fn < fu - 1e-12
            if not np.any(bad):
                break
            lam[bad] *= 0.5
        u, fu = un, fn
        if np.max(np.abs(lam[:, None] * step)) < 1e-8:
            break
    pe = expit(u[:, 0]); pf = expit(u[:, 1])
    h11 = n1 * pe * (1 - pe) + Si[0, 0]; h22 = n0 * pf * (1 - pf) + Si[1, 1]; h12 = Si[0, 1]
    # Cholesky factor of (-H)^{-1}
    dh = h11 * h22 - h12 * h12
    i11 = h22 / dh; i22 = h11 / dh; i12 = -h12 / dh
    c11 = np.sqrt(i11); c21 = i12 / c11; c22 = np.sqrt(np.maximum(i22 - c21 * c21, 1e-300))
    logdetC = np.log(c11) + np.log(c22)
    # quadrature nodes u_q = u_hat + C z_q, shape (k, Q^2, 2)
    e_q = u[:, None, 0] + c11[:, None] * Z[None, :, 0]
    f_q = u[:, None, 1] + c21[:, None] * Z[None, :, 0] + c22[:, None] * Z[None, :, 1]
    lp = (TP[:, None] * e_q - n1[:, None] * np.logaddexp(0, e_q)
          + FP[:, None] * f_q - n0[:, None] * np.logaddexp(0, f_q))
    de = e_q - m[:, None, 0]; df = f_q - m[:, None, 1]
    quad = Si[0, 0] * de * de + 2 * Si[0, 1] * de * df + Si[1, 1] * df * df
    lnorm = -0.5 * quad - 0.5 * np.log(det) - np.log(2 * np.pi)
    lf = lp + lnorm + Z2H[None, :]
    mx = lf.max(axis=1, keepdims=True)
    li = mx[:, 0] + np.log(np.sum(W[None, :] * np.exp(lf - mx), axis=1)) + logdetC + np.log(2 * np.pi)
    return -np.sum(li + const)


def _numerical_hessian(f, th, h=1e-4):
    p = len(th); H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ej = np.zeros(p); ei[i] = h; ej[j] = h
            H[i, j] = H[j, i] = (f(th + ei + ej) - f(th + ei - ej) - f(th - ei + ej) + f(th - ei - ej)) / (4 * h * h)
    return H


def default_start(TP, FN, FP, TN):
    eta = np.log((TP + .5) / (FN + .5)); phi = np.log((FP + .5) / (TN + .5))
    return np.array([eta.mean(), 0.0, phi.mean(), 0.0,
                     np.log(max(eta.std(), 0.2)), np.log(max(phi.std(), 0.2)),
                     np.arctanh(np.clip(np.corrcoef(eta, phi)[0, 1], -0.9, 0.9))])


def fit(TP, FN, FP, TN, x, start=None):
    """Full maximum-likelihood fit: L-BFGS-B followed by a Nelder-Mead polish, numerical
    Hessian for the observed information.  Used for the applications.  Returns theta, its
    covariance V, the negative log-likelihood and the parameter estimates on their natural
    scales."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    const = _binom_const(TP, n1, FP, n0)
    if start is None:
        start = default_start(TP, FN, FP, TN)
    f = lambda th: negloglik(th, TP, n1, FP, n0, x, const)
    start = np.clip(start, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    res = minimize(f, start, method="L-BFGS-B", bounds=BOUNDS,
                   options=dict(maxiter=500, ftol=1e-12, gtol=1e-7))
    res2 = minimize(f, res.x, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
    th = res2.x if res2.fun < res.fun else res.x
    th[4:6] = np.clip(th[4:6], -4, 3); th[6] = np.clip(th[6], -4, 4)
    f0 = f(th)
    H = _numerical_hessian(f, th)
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        V = np.full((7, 7), np.nan)
    mu_e, be, mu_f, bf, lse, lsf, t = th
    return dict(k=len(TP), mu_eta=float(mu_e), beta_eta=float(be), se_eta=float(np.sqrt(V[1, 1])),
                mu_phi=float(mu_f), beta_phi=float(bf), se_phi=float(np.sqrt(V[3, 3])),
                sig_e=float(np.exp(lse)), sig_f=float(np.exp(lsf)), rho=float(np.tanh(t)),
                theta=th, V=V, nll=float(f0), converged=bool(res.success or res2.success))


def start_from_normal(normal_fit):
    """Starting value for the likelihood fit from a normal-approximation fit (``normal.analyse``)."""
    f = normal_fit
    return np.array([f["beta"][0], f["beta"][1], f["beta"][2], f["beta"][3],
                     np.log(f["sig_e"]), np.log(f["sig_f"]),
                     np.arctanh(np.clip(f["rho"], -0.95, 0.95))])
