"""Data-generating mechanism of the simulation study (Section 3 and supplement Section C.1).

Study sizes first (``draw_sizes``: total size log-normal with median 300 and log-scale SD 0.8,
clipped to [40, 4000]; prevalence 0.35; at least 10 per group); z_i is the standardized
1/sqrt(ESS_i) within the replicate; then the latent threshold and the latent accuracy of the
HSROC model, with the size trends added on top of the residual variation, and binomial counts:

    theta_i = Theta + rho_s sig_theta z_i + sig_theta e_i,
    alpha_i = Lambda + delta z_i + sig_alpha e'_i,
    eta_i = lam^{1/2} (theta_i + alpha_i / 2),   phi_i = lam^{-1/2} (theta_i - alpha_i / 2),
    sig_eta sig_phi = 0.75^2,  sig_theta^2 = 0.75^2 (1 + rho) / 2,  sig_alpha^2 = 2 * 0.75^2 (1 - rho),

so that the shape lam, the residual correlation rho of the logits and the mean operating
point (mu_eta, mu_phi) = (1.0, -2.0) are exact design values in every setting.  rho_s is the
threshold trend in residual SDs of the latent threshold per SD of size; delta is the accuracy
trend in latent-accuracy units per SD of size.  The order of the random draws is part of the
specification: the seed of each setting (``analysis/simulation_settings.csv``) reproduces every
replicate exactly, and the fits consume no random numbers, so the replicate data sets can be
regenerated from the seeds alone.
"""
import numpy as np

MU_E, MU_F = 1.0, -2.0          # mean operating point on the logit scale
S2 = 0.75 ** 2                  # sig_eta * sig_phi
PREV = 0.35
E = lambda v: 1 / (1 + np.exp(-v))


def draw_sizes(k, rng, log_sd=0.8):
    N = np.clip(np.round(rng.lognormal(np.log(300), log_sd, k)), 40, 4000).astype(int)
    n1 = np.maximum((N * PREV).astype(int), 10)
    n0 = np.maximum(N - n1, 10)
    return n1, n0


def _sizes(k, rng):
    n1, n0 = draw_sizes(k, rng)
    s = 1 / np.sqrt(4 * n1 * n0 / (n1 + n0))
    zs = (s - s.mean()) / max(s.std(), 1e-9)
    return n1, n0, zs


def gen_H(k, lam, rho, rho_s, delta, rng, mu_e=MU_E, mu_f=MU_F):
    """One simulated review: TP, FN, FP, TN of k studies under the HSROC truth."""
    r = np.sqrt(lam)
    Th = (mu_e / r + mu_f * r) / 2
    La = mu_e / r - mu_f * r
    s_th = np.sqrt(S2 * (1 + rho) / 2); s_al = np.sqrt(2 * S2 * (1 - rho))
    n1, n0, zs = _sizes(k, rng)
    theta = Th + rho_s * s_th * zs + s_th * rng.normal(size=k)
    alpha = La + delta * zs + s_al * rng.normal(size=k)
    eta = r * (theta + alpha / 2); phi = (theta - alpha / 2) / r
    TP = rng.binomial(n1, E(eta)); FP = rng.binomial(n0, np.clip(E(phi), 1e-4, 1))
    return TP, n1 - TP, FP, n0 - FP
