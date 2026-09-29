#!/usr/bin/env python3
"""Maximum-count endpoint of the IPG review (ipg_max.csv) from ipg_min.csv.

extract_ipg.py writes both endpoints when the monograph text is available.  This script rebuilds the
maximum-count endpoint without the text: for every cohort of ipg_min.csv the printed values (proportion
and exact 95% interval, two decimals) are recovered from the minimum-count table, the admissible
candidate set is re-enumerated with hta_text.solutions, and the same rule as extract_ipg.py is applied
(hta_text.pair_candidates / endpoints): the candidate (count, size) pairs of each coordinate are ordered
by group size, then by count, and the first KEEP = 12 are kept; the sensitivity and specificity candidates
are paired under a prevalence between 0.02 and 0.95; the pair with the largest total group size is the
maximum-count endpoint (ties by the sensitivity group size).  Rounding ties (a proportion or endpoint
printed at exactly x.xx5) are resolved towards the alternative that reproduces the stored minimum-count
table.

Usage: python derive_ipg_max.py   (writes ../ipg_max.csv; about one minute)
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import beta
from hta_text import solutions, cp_ci, pair_candidates, endpoints, KEEP

TOL = 0.005 + 1e-9


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
                pairs = pair_candidates(s_se, s_sp)
                if endpoints(pairs)[0] == ((TP, n1), (TN, n0)):
                    found = pairs
                    break
            if found:
                break
        assert found is not None, f"cohort {r.cohort_id}: stored minimum not reproduced"
        n_amb += len(found) > 1
        (tp, nn1), (tn, nn0) = endpoints(found)[1]
        rows.append(dict(TP=tp, FN=nn1 - tp, TN=tn, FP=nn0 - tn, n1=nn1, n0=nn0))
    out = dmin.copy()
    for c in ("TP", "FN", "TN", "FP", "n1", "n0"):
        out[c] = [row[c] for row in rows]
    out.to_csv("../ipg_max.csv", index=False)
    print(f"ipg_max.csv: k={len(out)}, ambiguous cohorts {n_amb}, total N {int((out.n1 + out.n0).sum())} "
          f"(minimum-count total {int((dmin.n1 + dmin.n0).sum())})")


if __name__ == "__main__":
    main()
