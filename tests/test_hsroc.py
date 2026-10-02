"""Smoke and regression tests.  Run with `pytest -q` (under a minute: the regression tests refit both reviews, re-analyse
stored simulation replicates and check every stored rejection count against its replicate records)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsroc import data, design, fitting, funnel, glmm, normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))


def _fit(key):
    TP, FN, FP, TN = data.load_review(key)
    s, x = data.size_covariate(TP, FN, FP, TN)
    a = normal.analyse(TP, FN, FP, TN, "smoothed")
    o = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(a["fit"]))
    return TP, FN, FP, TN, x, o


def test_datasets_load():
    for key, k in (("FIT", 23), ("IPG", 42), ("IPG_max", 42)):
        TP, FN, FP, TN = data.load_review(key)
        assert len(TP) == k
        assert (TP >= 0).all() and (FN >= 0).all() and (FP >= 0).all() and (TN >= 0).all()
        assert ((TP + FN) > 0).all() and ((FP + TN) > 0).all()


def test_generator_reproducible_and_exact_design():
    a = design.gen_H(30, 0.5, 0.4, 0.4, 0.0, np.random.default_rng(81012))
    b = design.gen_H(30, 0.5, 0.4, 0.4, 0.0, np.random.default_rng(81012))
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    # large review, no trend: the residual shape and correlation of the logits are the design values
    rng = np.random.default_rng(7)
    n1, n0, zs = design._sizes(40000, rng)
    lam, rho = 0.5, 0.4
    r = np.sqrt(lam); s_th = np.sqrt(design.S2 * (1 + rho) / 2); s_al = np.sqrt(2 * design.S2 * (1 - rho))
    theta = s_th * rng.normal(size=40000); alpha = s_al * rng.normal(size=40000)
    eta = r * (theta + alpha / 2); phi = (theta - alpha / 2) / r
    assert abs(eta.std() / phi.std() - lam) < 0.02 and abs(np.corrcoef(eta, phi)[0, 1] - rho) < 0.02


def test_deeks_test_matches_direct_wls():
    TP, FN, FP, TN = design.gen_H(30, 1.0, 0.4, 0.0, 0.0, np.random.default_rng(1))
    eta, phi = data.logits(TP, FN, FP, TN); ess = data.ess(TP, FN, FP, TN)
    slope, p = funnel.deeks_test(eta - phi, ess)
    x = 1 / np.sqrt(ess); X = np.column_stack([np.ones_like(x), x]); W = np.diag(ess)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ (eta - phi))
    assert abs(beta[1] - slope) < 1e-10 and 0 <= p <= 1


def test_hsroc_parametrization_round_trip():
    par6 = np.array([1.0, -2.0, 0.7, np.log(0.6), np.log(0.8), np.arctanh(0.3)])
    th = fitting._to7(par6, "alpha")
    lam = np.exp(th[4] - th[5])
    assert abs(th[1] / np.sqrt(lam) - th[3] * np.sqrt(lam)) < 1e-12          # gamma_alpha = 0
    th = fitting._to7(par6, "theta")
    assert abs(0.5 * (th[1] / np.sqrt(lam) + th[3] * np.sqrt(lam))) < 1e-12  # gamma_theta = 0


def test_glmm_recovers_the_design_on_a_large_review():
    rng = np.random.default_rng(11)
    TP, FN, FP, TN = design.gen_H(150, 0.5, 0.4, 0.0, 0.0, rng)
    s, x = data.size_covariate(TP, FN, FP, TN)
    o = glmm.fit(TP, FN, FP, TN, x)
    q = fitting.hsroc_quantities(o["theta"], o["V"], len(TP), x=x)
    assert abs(o["mu_eta"] - 1.0) < 0.25 and abs(o["mu_phi"] + 2.0) < 0.25
    assert abs(q["lam"] - 0.5) < 0.2 and abs(q["rho"] - 0.4) < 0.3
    assert q["lam_ci"][0] < q["lam"] < q["lam_ci"][1]


def test_fast_fit_and_null_fits_are_consistent():
    TP, FN, FP, TN = design.gen_H(30, 0.5, 0.4, 0.4, 0.0, np.random.default_rng(3))
    s, x = data.size_covariate(TP, FN, FP, TN)
    a = normal.analyse(TP, FN, FP, TN, "smoothed")
    g = fitting.fit_fast(TP, FN, FP, TN, x, glmm.start_from_normal(a["fit"]))
    assert 0 <= g["p_hs"] <= 1 and 0 <= g["p_c1"] <= 1 and g["c_h"] > 0
    for which in ("alpha", "theta"):
        nul = fitting.fit_null(TP, FN, FP, TN, x, g["theta"], which)
        assert nul["nll"] >= g["nll"] - 1e-6                  # the constrained optimum cannot beat the full one
        lr = fitting.lrt(g["nll"], nul["nll"], k=len(TP))
        assert lr["stat"] >= 0 and 0 <= lr["p"] <= 1 and 0 <= lr["p_t"] <= 1
        assert lr["p_t"] >= lr["p"] - 1e-12                   # the t reference is the more conservative one


def test_sauc_monotone_in_alpha():
    assert fitting.sauc(1.0, 0.0) == pytest.approx(0.5, abs=1e-3)
    assert fitting.sauc(0.7, 2.0) < fitting.sauc(0.7, 3.0)


def test_simulation_replicates_reproduce_the_stored_p_values():
    """The first replicates of a stored setting are regenerated from its seed and re-analysed: the data sets and the
    p values of every procedure equal the stored ones (results/simulation/<cell>.npz)."""
    import simulate as S
    cell = "K30-r0.4-l0.5-rs0.4-d0.0"
    path = ROOT / "results" / "simulation" / f"{cell}.npz"
    if not path.exists():
        pytest.skip(f"results/simulation/{cell}.npz not present")
    with np.load(path, allow_pickle=True) as zf:
        z = {name: zf[name] for name in zf.files}
    man = pd.read_csv(ROOT / "analysis" / "simulation_settings.csv", dtype={"lam_tag": str}).set_index("cell").loc[cell]
    k, lam = int(man["k"]), S.LAM_TAGS[str(man["lam_tag"])]
    rng = np.random.default_rng(int(man["seed"]))
    for r in range(3):
        TP, FN, FP, TN = design.gen_H(k, lam, float(man["rho"]), float(man["rho_s"]), float(man["delta"]), rng, design.MU_E, design.MU_F)
        assert np.array_equal(TP, z["TP"][r]) and np.array_equal(FP, z["FP"][r])
        rec = S.analyse_replicate(TP, FN, FP, TN, k)
        for name in ("p_deeks", "p_c1", "p_hsW", "p_hsL", "p_hsLt"):
            assert abs(rec[name] - float(z[name][r])) < 1e-9, (name, r)


def test_stored_rates_are_consistent_with_the_replicate_records():
    """For every stored setting, the rejection rate of each procedure in <cell>.json equals the rate recomputed from
    the replicate-level p values in <cell>.npz (valid replicates as denominator)."""
    files = sorted(glob.glob(str(ROOT / "results" / "simulation" / "*.json")))
    if not files:
        pytest.skip("results/simulation not present")
    import simulate as S
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        with np.load(f[:-5] + ".npz", allow_pickle=True) as z:
            recs = pd.DataFrame({c: z[c] for c in S.FAILED if c in z.files})
        valid = pd.DataFrame([S.valid_flags(r) for r in recs.to_dict("records")])
        for m, pcol in (("deeks", "p_deeks"), ("c1", "p_c1"), ("hsW", "p_hsW"), ("hsL", "p_hsL"), ("hsLt", "p_hsLt")):
            p = recs[pcol].values.astype(float); v = valid[m].values
            assert int(v.sum()) == j["methods"][m]["n_valid"], (j["cell"], m)
            assert int(np.sum((p < S.ALPHA) & v)) == j["methods"][m]["n_reject"], (j["cell"], m)


def test_applications_regression_against_stored_results():
    """results/applications.json is reproduced by a fresh fit of both reviews (shape, latent trends, tests)."""
    path = ROOT / "results" / "applications.json"
    if not path.exists():
        pytest.skip("results/applications.json not present")
    with open(path) as fh:
        stored = json.load(fh)
    for key in ("FIT", "IPG"):
        TP, FN, FP, TN, x, o = _fit(key)
        k = len(TP)
        q = fitting.hsroc_quantities(o["theta"], o["V"], k, x=x)
        nul = fitting.fit_null(TP, FN, FP, TN, x, o["theta"], "alpha")
        lr = fitting.lrt(o["nll"], nul["nll"], k=k)
        eta, phi = data.logits(TP, FN, FP, TN)
        slope, p_deeks = funnel.deeks_test(eta - phi, data.ess(TP, FN, FP, TN))
        r = stored[key]
        assert r["k"] == k
        assert abs(q["lam"] - r["lam"]) < 1e-6
        assert abs(q["gamma_alpha"]["est"] - r["gamma_alpha"]["est"]) < 1e-6
        assert abs(q["gamma_theta"]["est"] - r["gamma_theta"]["est"]) < 1e-6
        assert abs(lr["p_t"] - r["lrt_alpha"]["p_t"]) < 1e-6
        assert abs(slope - r["deeks_slope"]) < 1e-9 and abs(p_deeks - r["p_deeks"]) < 1e-9
        # Hessian-based quantities: the stored values use glmm.HESSIAN_STEP_APPLICATIONS (the default of glmm.fit)
        assert abs(q["betaA_1"]["se"] - r["betaA_1"]["se"]) < 1e-6 and abs(q["betaA_1"]["p"] - r["betaA_1"]["p"]) < 1e-6
        assert abs(q["gamma_alpha"]["se"] - r["gamma_alpha"]["se"]) < 1e-6


def test_hessian_step_of_the_applications_is_in_the_stable_range():
    """For the FIT review the standard error of the lnDOR trend of the binomial fit is the same (to 0.1%) at the
    application step 1e-3 and at 3e-3, i.e. the step is not in the noise-dominated range (which starts near 1e-4)."""
    TP, FN, FP, TN, x, o = _fit("FIT")
    n1 = TP + FN; n0 = FP + TN; const = glmm._binom_const(TP, n1, FP, n0)
    f = lambda th: glmm.negloglik(th, TP, n1, FP, n0, x, const)
    c = np.zeros(7); c[1] = 1.0; c[3] = -1.0
    se = {h: float(np.sqrt(c @ np.linalg.inv(glmm._numerical_hessian(f, o["theta"], h=h)) @ c)) for h in (1e-3, 3e-3)}
    assert abs(se[1e-3] / se[3e-3] - 1) < 1e-3
    assert glmm.HESSIAN_STEP_APPLICATIONS == 1e-3 and glmm.HESSIAN_STEP_SIMULATION == 1e-4


def test_delta_method_quantities_are_nan_not_floored_when_the_covariance_is_not_positive_definite():
    """A covariance matrix that is not positive definite (here the negated covariance of the FIT fit, so every
    delta-method variance is negative) gives NaN standard errors, p values and intervals with a RuntimeWarning,
    rather than a floored standard error and a spuriously small p value."""
    TP, FN, FP, TN, x, o = _fit("FIT")
    k = len(TP)
    with pytest.warns(RuntimeWarning, match="non-positive or non-finite"):
        q = fitting.hsroc_quantities(o["theta"], -o["V"], k)
    for name in ("gamma_alpha", "gamma_theta", "betaA_H", "betaA_1", "shape"):
        assert np.isnan(q[name]["se"]) and np.isnan(q[name]["p"]) and all(np.isnan(v) for v in q[name]["ci"])
        assert np.isfinite(q[name]["est"])
    assert np.isnan(q["lam_se"]) and np.isfinite(q["lam"])
    # the fitted FIT review itself has a positive-definite Hessian and finite standard errors
    assert o["hess_pd"] and np.isfinite(fitting.hsroc_quantities(o["theta"], o["V"], k)["gamma_alpha"]["se"])
