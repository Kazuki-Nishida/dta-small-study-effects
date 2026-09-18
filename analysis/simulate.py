#!/usr/bin/env python3
"""The simulation study (Section 3 of the paper; supplement Section C): the three procedures of the paper and two
auxiliary test forms at the 80 settings of analysis/simulation_settings.csv, with replicate-level records.

Settings (one row of the manifest each; the seed of every setting is recorded there):
  shape               k = 30, rho = 0.4; lambda in {1/4, 1/2, 1/sqrt2, 1, sqrt2, 2, 4} under no trend, a threshold trend
                      alone (rho_s = 0.4), an accuracy trend alone (delta = 0.5) and both trends -- 28 settings
                      (Figures 1(a), 2(a), 2(b); Tables S1 and S4)
  threshold-strength  k = 30, rho = 0.4, lambda = 1/2, delta = 0; rho_s in {0.2, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6} -- 7 settings
                      (with the two shape settings at rho_s = 0 and 0.4: Figure 1(b); Tables S2 and S4)
  k-comparison        rho = 0.4; k in {10, 20, 50}; lambda in {1/2, 1, 2}; no trend, threshold trend, accuracy trend
                      -- 27 settings (Table S3; Tables S5 to S7)
  rho-comparison      k = 30; rho in {0, 0.8}; lambda in {1/2, 1, 2}; the same three scenarios -- 18 settings
                      (Table S3; Table S7)

Generator: hsroc.design.gen_H -- total size log-normal (median 300, log-SD 0.8), rounded, clipped to [40, 4000];
n1 = max(floor(0.35 N), 10), n0 = max(N - n1, 10); ESS = 4 n1 n0 / (n1 + n0); s = ESS^{-1/2}; z = (s - mean s) / SD(s)
with the population SD; sig_eta sig_phi = 0.75^2, mu_eta = 1, mu_phi = -2; the latent draws in the order theta, alpha;
binomial counts with expit(phi) clipped below at 1e-4.  The fits use x = s - mean(s), so the fitted trends relate to
the generating ones by gamma_alpha * SD(s) = delta and gamma_theta * SD(s) = rho_s sigma_theta.

Procedures (rejection at the 0.10 level, two-sided):
  deeks   WLS of the continuity-corrected empirical lnDOR (+0.5 in every cell) on s, weights ESS, t_{k-2}
  c1      lnDOR trend of the binomial fit, beta_eta - beta_phi, delta-method SE, t_{k-2}
  hsLt    the proposed test: likelihood ratio for gamma_alpha = 0 with the threshold trend free and all other parameters
          re-estimated, signed root referred to t_{k-2}
  hsW     auxiliary: Wald contrast of the accuracy coordinate beta_eta - lam-hat beta_phi (delta method including lam-hat), t_{k-2}
  hsL     auxiliary: the likelihood-ratio statistic of hsLt referred to chi^2_1
Fits: L-BFGS-B (maxiter 500, ftol 1e-12, gtol 1e-6) from the smoothed-variance normal-approximation start, bounds of
hsroc.glmm (log sigma in [-4, 3], artanh rho in [-4, 4]); numerical Hessian (h = 1e-4); the constrained fit with two
starts (hsroc.fitting.fit_null).  No Nelder-Mead polish and no retries.

Recorded per replicate (results/simulation/<cell>.npz): the four counts of every study, SD(s), nll_full, nll_null,
convergence flags, boundary flags (full and constrained fit), Hessian invertible / covariance finite, validity of the
two Wald variances, the raw likelihood ratio (before clipping at 0), the five p values and the fitted lam-hat.
Replicates with an optimizer warning are kept; a p value is "valid" when it is finite and, for the Wald contrasts,
when the variance used is finite and positive; rejection rates are reported with the valid replicates as denominator,
and also with all generated replicates (an invalid p value counted as no rejection) for comparison.  The per-setting
summary is results/simulation/<cell>.json; analysis/collect_simulation.py gathers the summaries into one table.

Usage: python analysis/simulate.py --cells 0:80 [--reps 1000] [--out results/simulation]
       python analysis/simulate.py --list
Settings already summarised in the output folder are skipped (resumable).  Long runs can be split into time-budgeted
chunks with --chunk CELL --budget SECONDS (part files, merged when complete; the random stream is identical)."""
import os, sys, time, json, argparse
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, t as tdist

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from hsroc import glmm, fitting
from hsroc.design import gen_H, MU_E, MU_F, S2
from hsroc.normal import analyse
from hsroc.funnel import deeks_test

ALPHA = 0.10
LAM_TAGS = {"0.25": 0.25, "0.5": 0.5, "0.71": 2 ** -0.5, "1.0": 1.0, "1.41": 2 ** 0.5, "2.0": 2.0, "4.0": 4.0}
METHODS = ["deeks", "c1", "hsW", "hsL", "hsLt"]
MANIFEST = os.path.join(HERE, "simulation_settings.csv")


# ------------------------------------------------------------------ design quantities
def design_quantities(lam, rho, rho_s, delta):
    s_th = np.sqrt(S2 * (1 + rho) / 2); s_al = np.sqrt(2 * S2 * (1 - rho))
    b_dor = lam ** -0.5 * ((lam - 1) * rho_s * s_th + (lam + 1) / 2 * delta)   # true lnDOR slope per SD of size (z)
    return dict(sigma_alpha=float(s_al), sigma_theta=float(s_th), delta_over_sigma_alpha=float(delta / s_al) if delta else 0.0,
                b_dor=float(b_dor), deeks_weight=float(lam ** -0.5 * (lam - 1)))


# ------------------------------------------------------------------ the full fit with diagnostics (the algorithm of fitting.fit_fast)
def fit_full_record(TP, FN, FP, TN, x, start):
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN; const = glmm._binom_const(TP, n1, FP, n0)
    f = lambda th: glmm.negloglik(th, TP, n1, FP, n0, x, const)
    B = glmm.BOUNDS; lo = np.array([b[0] for b in B]); hi = np.array([b[1] for b in B])
    start = np.clip(start, lo, hi)
    res = minimize(f, start, method="L-BFGS-B", bounds=B, options=dict(maxiter=500, ftol=1e-12, gtol=1e-6))
    th = res.x.copy(); p = 7
    H = glmm._numerical_hessian(f, th)
    hess_ok = True
    try:
        V = np.linalg.inv(H)
        if not np.all(np.isfinite(V)):
            hess_ok = False
    except np.linalg.LinAlgError:
        V = np.full((p, p), np.nan); hess_ok = False
    mu_e, be, mu_f, bf, lse, lsf, t = th
    c_h = float(np.exp(lse) / np.exp(lsf)); gc = np.zeros(p); gc[4] = c_h; gc[5] = -c_h
    k = len(TP)

    def contrast(c, gvec):
        g = np.zeros(p); g[1] = 1.0; g[3] = -c; g = g - bf * gvec
        est = be - c * bf
        var = float(g @ V @ g) if hess_ok else float("nan")
        ok = bool(np.isfinite(var) and var > 1e-12)
        se_ = np.sqrt(max(var, 1e-12)) if np.isfinite(var) else float("nan")
        pval = float(2 * tdist.sf(abs(est / se_), k - 2)) if np.isfinite(se_) else float("nan")
        return pval, ok, est, var
    p_hs, ok_hs, est_hs, var_hs = contrast(c_h, gc); p_c1, ok_c1, est_c1, var_c1 = contrast(1.0, np.zeros(p))
    tol = 1e-6
    bound = dict(lse=bool(abs(lse - lo[4]) < tol or abs(lse - hi[4]) < tol), lsf=bool(abs(lsf - lo[5]) < tol or abs(lsf - hi[5]) < tol),
                 t=bool(abs(t - lo[6]) < tol or abs(t - hi[6]) < tol), beta=bool(np.any(np.abs(th[[1, 3]] - lo[[1, 3]]) < tol) or np.any(np.abs(th[[1, 3]] - hi[[1, 3]]) < tol)))
    return dict(theta=th, nll=float(res.fun), converged=bool(res.success), status=int(res.status), nit=int(res.nit), hess_ok=hess_ok,
                p_hs=p_hs, ok_hs=ok_hs, p_c1=p_c1, ok_c1=ok_c1, lam_hat=c_h, bound=bound, est_hs=est_hs, var_hs=var_hs, est_c1=est_c1, var_c1=var_c1)


def analyse_replicate(TP, FN, FP, TN, k):
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    eta = np.log((TP + .5) / (FN + .5)); phi = np.log((FP + .5) / (TN + .5))
    n1 = TP + FN; n0 = FP + TN; ess = 4 * n1 * n0 / (n1 + n0); s = 1 / np.sqrt(ess); x = s - s.mean()
    a = analyse(TP, FN, FP, TN, "smoothed"); f = a["fit"]
    start = np.array([f["beta"][0], f["beta"][1], f["beta"][2], f["beta"][3],
                      np.log(f["sig_e"]), np.log(f["sig_f"]), np.arctanh(np.clip(f["rho"], -0.95, 0.95))])
    g = fit_full_record(TP, FN, FP, TN, x, start)
    nul = fitting.fit_null(TP, FN, FP, TN, x, g["theta"], "alpha")
    lr_raw = 2.0 * (nul["nll"] - g["nll"]); lr = max(lr_raw, 0.0)
    p_hsL = float(chi2.sf(lr, 1)); p_hsLt = float(2 * tdist.sf(np.sqrt(lr), k - 2))
    th0 = nul["theta7"]; tol = 1e-6
    bound_null = bool(abs(th0[4] + 4) < tol or abs(th0[4] - 3) < tol or abs(th0[5] + 4) < tol or abs(th0[5] - 3) < tol or abs(abs(th0[6]) - 4) < tol)
    slope, p_deeks = deeks_test(eta - phi, ess)
    rec = dict(s_sd=float(s.std()), nll_full=g["nll"], nll_null=nul["nll"], conv_full=g["converged"], conv_null=nul["converged"],
               status_full=g["status"], nit_full=g["nit"], hess_ok=g["hess_ok"], ok_hs=g["ok_hs"], ok_c1=g["ok_c1"],
               bound_lse=g["bound"]["lse"], bound_lsf=g["bound"]["lsf"], bound_t=g["bound"]["t"], bound_beta=g["bound"]["beta"], bound_null=bound_null,
               lr_raw=float(lr_raw), lam_hat=g["lam_hat"], deeks_slope=float(slope), est_hs=g["est_hs"], est_c1=g["est_c1"],
               p_deeks=float(p_deeks), p_c1=g["p_c1"], p_hsW=g["p_hs"], p_hsL=p_hsL, p_hsLt=p_hsLt)
    return rec


def valid_flags(rec):
    """Validity of each p value: finite, and for the Wald contrasts a finite positive variance."""
    return dict(deeks=bool(np.isfinite(rec["p_deeks"])), c1=bool(np.isfinite(rec["p_c1"]) and rec["ok_c1"]),
                hsW=bool(np.isfinite(rec["p_hsW"]) and rec["ok_hs"]), hsL=bool(np.isfinite(rec["p_hsL"])), hsLt=bool(np.isfinite(rec["p_hsLt"])))


FAILED = dict(s_sd=float("nan"), nll_full=float("nan"), nll_null=float("nan"), conv_full=False, conv_null=False, status_full=-1, nit_full=0,
              hess_ok=False, ok_hs=False, ok_c1=False, bound_lse=False, bound_lsf=False, bound_t=False, bound_beta=False, bound_null=False,
              lr_raw=float("nan"), lam_hat=float("nan"), deeks_slope=float("nan"), est_hs=float("nan"), est_c1=float("nan"),
              p_deeks=float("nan"), p_c1=float("nan"), p_hsW=float("nan"), p_hsL=float("nan"), p_hsLt=float("nan"))


def cell_params(row):
    return dict(tag=row["cell"], k=int(row["k"]), rho=float(row["rho"]), lam=LAM_TAGS[str(row["lam_tag"])], rho_s=float(row["rho_s"]),
                delta=float(row["delta"]), seed=int(row["seed"]))


def finish_cell(df, data, row, lam, reps, seed, out_dir, t0, log):
    tag = row["cell"]
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), cell=tag, k=int(row["k"]), rho=float(row["rho"]), lam=lam, rho_s=float(row["rho_s"]),
                        delta=float(row["delta"]), seed=seed, reps=reps, **{nm: np.array(data[nm]) for nm in data}, **{c: df[c].values for c in df.columns if c != "error"})
    summ = summarise(df, row, lam, reps, seed)
    summ["seconds"] = round(time.time() - t0); summ["numpy"] = np.__version__
    import scipy; summ["scipy"] = scipy.__version__
    json.dump(summ, open(os.path.join(out_dir, f"{tag}.json"), "w"), indent=1)
    log(f"{tag:34s} | " + " ".join(f"{m} {summ['methods'][m]['rate']:.3f}" for m in METHODS) +
        f" | lam-hat {summ['mean_lam_hat']:.2f} nonconv {summ['nonconverged_full']}/{summ['nonconverged_null']} bound {summ['boundary_full']} "
        f"negLR {summ['negative_lr']} invalid {summ['invalid_any']} [{summ['seconds']}s]")
    return summ


# ------------------------------------------------------------------ chunked execution (time-budgeted partial runs)
def run_chunk(row, reps, out_dir, start, budget, log=print):
    """Replicates start..(as many as fit in `budget` seconds) of one setting, written to <cell>.part_<start>-<last>.npz.
    The random stream is identical to the sequential run: replicates before `start` are regenerated without fitting
    (the fits consume no random numbers)."""
    P = cell_params(row); tag = P["tag"]; k = P["k"]; lam = P["lam"]
    rng = np.random.default_rng(P["seed"]); t0 = time.time()
    for r in range(start - 1):
        gen_H(k, lam, P["rho"], P["rho_s"], P["delta"], rng, MU_E, MU_F)
    recs = []; data = dict(TP=[], FN=[], FP=[], TN=[]); r = start
    while r <= reps and (time.time() - t0) < budget:
        TP, FN, FP, TN = gen_H(k, lam, P["rho"], P["rho_s"], P["delta"], rng, MU_E, MU_F)
        try:
            rec = analyse_replicate(TP, FN, FP, TN, k)
        except Exception as ex:
            rec = dict(FAILED); rec["error"] = str(ex)[:80]
        rec["rep"] = r; recs.append(rec)
        for nm, v in zip(("TP", "FN", "FP", "TN"), (TP, FN, FP, TN)):
            data[nm].append(np.asarray(v, int))
        r += 1
    if not recs:
        return start - 1
    df = pd.DataFrame(recs); last = int(df["rep"].max())
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{tag}.part_{start}-{last}.npz"), cell=tag, seed=P["seed"],
                        **{nm: np.array(data[nm]) for nm in data}, **{c: df[c].values for c in df.columns if c != "error"})
    log(f"chunk {tag} {start}-{last} [{time.time() - t0:.0f}s]")
    return last


def merge_parts(row, reps, out_dir, log=print):
    """Assemble the part files of a setting into the standard <cell>.npz / <cell>.json (if the replicates 1..reps are complete)."""
    import glob, re
    P = cell_params(row); tag = P["tag"]
    parts = sorted(glob.glob(os.path.join(out_dir, f"{tag}.part_*.npz")), key=lambda f: int(re.search(r"part_(\d+)-", f).group(1)))
    if not parts:
        return False
    frames = []; data = dict(TP=[], FN=[], FP=[], TN=[])
    cols = [c for c in FAILED] + ["rep"]
    for f in parts:
        z = np.load(f, allow_pickle=True)
        d = {c: z[c] for c in cols if c in z.files}; frames.append(pd.DataFrame(d))
        for nm in data:
            data[nm].extend(list(z[nm]))
    df = pd.concat(frames).drop_duplicates("rep").sort_values("rep").reset_index(drop=True)
    have = set(df["rep"].astype(int)); missing = [r for r in range(1, reps + 1) if r not in have]
    if missing:
        log(f"{tag}: {len(have)} replicates in parts, missing {len(missing)} (next start {missing[0]})"); return False
    # data rows in the same order as df (parts may overlap after a resume)
    reps_in_parts = np.concatenate([np.load(f, allow_pickle=True)["rep"] for f in parts]).astype(int)
    order = {}
    for i, rp in enumerate(reps_in_parts):
        order.setdefault(int(rp), i)
    idx = [order[r] for r in range(1, reps + 1)]
    data = {nm: [data[nm][i] for i in idx] for nm in data}
    finish_cell(df, data, row, P["lam"], reps, P["seed"], out_dir, time.time(), log)
    return True


def next_start(row, reps, out_dir):
    """First replicate not yet covered by a part file (reps + 1 when complete)."""
    import glob, re
    have = set()
    for f in glob.glob(os.path.join(out_dir, f"{row['cell']}.part_*.npz")):
        a, b = map(int, re.search(r"part_(\d+)-(\d+)", f).groups()); have.update(range(a, b + 1))
    r = 1
    while r in have:
        r += 1
    return r


# ------------------------------------------------------------------ one setting
def run_cell(row, reps, out_dir, log=print):
    tag = row["cell"]; k = int(row["k"]); rho = float(row["rho"]); lam = LAM_TAGS[str(row["lam_tag"])]
    rho_s = float(row["rho_s"]); delta = float(row["delta"]); seed = int(row["seed"])
    rng = np.random.default_rng(seed)
    recs = []; data = dict(TP=[], FN=[], FP=[], TN=[]); t0 = time.time()
    for r in range(reps):
        TP, FN, FP, TN = gen_H(k, lam, rho, rho_s, delta, rng, MU_E, MU_F)
        try:
            rec = analyse_replicate(TP, FN, FP, TN, k)
        except Exception as ex:   # a failed replicate: recorded as invalid for every method
            rec = dict(FAILED); rec["error"] = str(ex)[:80]
        rec["rep"] = r + 1; recs.append(rec)
        for nm, v in zip(("TP", "FN", "FP", "TN"), (TP, FN, FP, TN)):
            data[nm].append(np.asarray(v, int))
        if (r + 1) % 250 == 0:
            log(f"  {tag}: {r + 1}/{reps} [{time.time() - t0:.0f}s]")
    df = pd.DataFrame(recs)
    return finish_cell(df, data, row, lam, reps, seed, out_dir, t0, log)


def summarise(df, row, lam, reps, seed):
    k = int(row["k"]); rho = float(row["rho"]); rho_s = float(row["rho_s"]); delta = float(row["delta"])
    dq = design_quantities(lam, rho, rho_s, delta)
    pcol = dict(deeks="p_deeks", c1="p_c1", hsW="p_hsW", hsL="p_hsL", hsLt="p_hsLt")
    methods = {}
    valid = pd.DataFrame([valid_flags(r) for r in df.to_dict("records")])
    for m in METHODS:
        p = df[pcol[m]].values.astype(float); v = valid[m].values
        n_valid = int(v.sum()); n_rej = int(np.sum((p < ALPHA) & v)); rate = n_rej / n_valid if n_valid else float("nan")
        methods[m] = dict(n_generated=int(len(df)), n_valid=n_valid, n_invalid=int(len(df) - n_valid), n_reject=n_rej, rate=float(rate),
                          rate_all_denominator=float(np.sum(np.nan_to_num(p, nan=1.0) < ALPHA) / len(df)),
                          mcse=float(np.sqrt(rate * (1 - rate) / n_valid)) if n_valid else float("nan"))
    neg = df["lr_raw"].values.astype(float); neg_lr = int(np.sum(neg < -1e-6)); min_lr = float(np.nanmin(neg)) if np.isfinite(neg).any() else float("nan")
    out = dict(cell=row["cell"], block=row.get("block", ""), k=k, rho=rho, lam=float(lam), lam_tag=str(row["lam_tag"]),
               rho_s=rho_s, delta=delta, seed=seed, reps=int(reps), **dq, mean_s_sd=float(np.nanmean(df["s_sd"])),
               nonconverged_full=int((~df["conv_full"].astype(bool)).sum()), nonconverged_null=int((~df["conv_null"].astype(bool)).sum()),
               boundary_full=int((df["bound_lse"] | df["bound_lsf"] | df["bound_t"]).sum()), boundary_t=int(df["bound_t"].sum()),
               boundary_sig=int((df["bound_lse"] | df["bound_lsf"]).sum()), boundary_beta=int(df["bound_beta"].sum()), boundary_null=int(df["bound_null"].sum()),
               hessian_not_invertible=int((~df["hess_ok"].astype(bool)).sum()), wald_hs_invalid=int((~df["ok_hs"].astype(bool)).sum()),
               wald_c1_invalid=int((~df["ok_c1"].astype(bool)).sum()), negative_lr=neg_lr, min_lr_raw=min_lr,
               failed_replicates=int(df["error"].notna().sum()) if "error" in df else 0,
               invalid_any=int((~valid.all(axis=1)).sum()), mean_lam_hat=float(np.nanmean(df["lam_hat"])), median_lam_hat=float(np.nanmedian(df["lam_hat"])),
               methods=methods)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="index range a:b or comma list into the manifest (default: all)")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "simulation"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--chunk", default="", help="cell name: run a time-budgeted chunk of replicates (resumes from the part files)")
    ap.add_argument("--budget", type=float, default=160.0, help="seconds of fitting for --chunk")
    ap.add_argument("--merge", action="store_true", help="assemble complete part files into <cell>.npz/.json for the selected settings")
    ap.add_argument("--next", action="store_true", help="print the unfinished settings of the selection with their next replicate")
    args = ap.parse_args()
    m = pd.read_csv(MANIFEST, dtype={"lam_tag": str})
    if args.list:
        print(m.to_string()); return
    idx = list(range(len(m)))
    if args.cells and ":" in args.cells:
        a, b = args.cells.split(":"); idx = list(range(int(a) if a else 0, int(b) if b else len(m)))
    elif args.cells:
        idx = [int(v) for v in args.cells.split(",")]
    os.makedirs(args.out, exist_ok=True); logpath = os.path.join(args.out, f"run_{os.getpid()}.log")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"; print(line, flush=True)
        with open(logpath, "a") as fh:
            fh.write(line + "\n")
    if args.chunk:
        row = m[m["cell"] == args.chunk].iloc[0].to_dict()
        if os.path.exists(os.path.join(args.out, f"{row['cell']}.json")):
            print(f"{row['cell']} done"); return
        st = next_start(row, args.reps, args.out)
        if st <= args.reps:
            run_chunk(row, args.reps, args.out, st, args.budget, log=log)
        merge_parts(row, args.reps, args.out, log=log); return
    if args.next:
        for i in idx:
            row = m.iloc[i].to_dict()
            if not os.path.exists(os.path.join(args.out, f"{row['cell']}.json")):
                print(i, row["cell"], next_start(row, args.reps, args.out))
        return
    for i in idx:
        row = m.iloc[i].to_dict()
        if os.path.exists(os.path.join(args.out, f"{row['cell']}.json")):
            log(f"skip {row['cell']} (done)"); continue
        if args.merge:
            merge_parts(row, args.reps, args.out, log=log); continue
        run_cell(row, args.reps, args.out, log=log)
    log("finished")


if __name__ == "__main__":
    main()
