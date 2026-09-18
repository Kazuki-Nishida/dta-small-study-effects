#!/usr/bin/env python3
"""Maximum-count endpoint of the IPG review (ipg_max.csv) from ipg_min.csv.

extract_ipg.py writes both endpoints when the monograph text is available.  This script
rebuilds the maximum-count endpoint without the text: for every cohort of ipg_min.csv the
printed values (proportion and exact 95% interval, two decimals) are recovered from the
minimum-count table, the admissible candidate set is re-enumerated with the same tolerance
rule as hta_text.solutions, truncated to its twelve smallest sample sizes per
coordinate (which keeps the "maximum" finite for the high-accuracy cohorts whose printed
precision hardly bounds the size), paired under the same prevalence filter as extract_ipg.py,
and the largest-count pair is written.  Rounding ties (a proportion or endpoint printed at exactly x.xx5) are resolved
towards the alternative that reproduces the stored minimum-count table.

Usage: python derive_ipg_max.py   (writes ../ipg_max.csv; about one minute)
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import beta

TOL = 0.005 + 1e-9
KEEP = 12            # candidates kept per coordinate (smallest sample sizes)
PREV_LO, PREV_HI = 0.02, 0.95   # pairing filter of extract_ipg.py


def cp_ci(x, n, a=0.05):
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


def solutions(phat, lo, hi, nmax=4000):
    """All (x, n) whose proportion and exact interval round to the printed values (vectorised)."""
    n = np.arange(1, nmax + 1)
    cands = set()
    base = np.round(phat * n).astype(int)
    for x in (base - 1, base, base + 1, np.floor(phat * n).astype(int), np.ceil(phat * n).astype(int)):
        ok = (x >= 0) & (x <= n) & (np.abs(x / n - phat) <= TOL)
        xs, ns = x[ok], n[ok]
        with np.errstate(all="ignore"):
            l = np.where(xs == 0, 0.0, beta.ppf(0.025, np.maximum(xs, 1), ns - xs + 1))
            h = np.where(xs == ns, 1.0, beta.ppf(0.975, xs + 1, np.maximum(ns - xs, 1)))
        m = (np.abs(l - lo) <= TOL) & (np.abs(h - hi) <= TOL)
        cands.update(zip(xs[m].tolist(), ns[m].tolist()))
    return sorted(cands, key=lambda s: s[1])


def printed_alternatives(x, n):
    """Printed (p, lo, hi) implied by a count; two alternatives per value at a rounding tie."""
    vals = [x / n, *cp_ci(x, n)]
    opts = []
    for v in vals:
        if abs((v * 100) % 1 - 0.5) < 1e-6:
            opts.append(sorted({round(v - 0.005 + 1e-9, 2), round(v + 0.005 - 1e-9, 2)}))
        else:
            opts.append([round(v + 1e-12, 2)])
    return list(itertools.product(*opts))


def main():
    dmin = pd.read_csv("../ipg_min.csv")
    rows, n_amb = [], 0
    for _, r in dmin.iterrows():
        TP, FN, FP, TN = int(r.TP), int(r.FN), int(r.FP), int(r.TN)
        n1, n0 = TP + FN, FP + TN
        found = None
        for pse in printed_alternatives(TP, n1):
            for psp in printed_alternatives(TN, n0):
                s_se, s_sp = solutions(*pse)[:KEEP], solutions(*psp)[:KEEP]
                if (TP, n1) not in s_se or (TN, n0) not in s_sp:
                    continue
                pairs = [(a, b) for a in s_se for b in s_sp if PREV_LO <= a[1] / (a[1] + b[1]) <= PREV_HI]
                pairs = pairs or [(a, b) for a in s_se for b in s_sp]
                if min(pairs, key=lambda p: (p[0][1] + p[1][1], p[0][1])) == ((TP, n1), (TN, n0)):
                    found = pairs
                    break
            if found:
                break
        assert found is not None, f"cohort {r.cohort_id}: stored minimum not reproduced"
        n_amb += len(found) > 1
        (tp, nn1), (tn, nn0) = max(found, key=lambda p: (p[0][1] + p[1][1], p[0][1]))
        rows.append(dict(TP=tp, FN=nn1 - tp, TN=tn, FP=nn0 - tn, n1=nn1, n0=nn0))
    out = dmin.copy()
    for c in ("TP", "FN", "TN", "FP", "n1", "n0"):
        out[c] = [row[c] for row in rows]
    out.to_csv("../ipg_max.csv", index=False)
    print(f"ipg_max.csv: k={len(out)}, ambiguous cohorts {n_amb}, total N {int((out.n1 + out.n0).sum())} "
          f"(minimum-count total {int((dmin.n1 + dmin.n0).sum())})")


if __name__ == "__main__":
    main()
