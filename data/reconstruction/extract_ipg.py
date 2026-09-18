#!/usr/bin/env python3
"""Extract the IPG (symptomatic DVT) meta-analysis from Goodacre et al., HTA 2006;10(15):
Figures 69 (Se) / 70 (Sp), k = 42; years from the reference list; 2 x 2 tables by
Clopper-Pearson inversion; the minimum-count and maximum-count data sets (ipg_min.csv, ipg_max.csv).
Input: hta1015.txt and hta1015_raw.txt (see README.md); not distributed."""
import re
import numpy as np
import pandas as pd
from hta_text import solutions, load_years, cp_ci

TXT = "hta1015.txt"
ROW = re.compile(r"^\s*(.+?)(\d{2,3})\s+([01][.,]\d{2})\s+\(([01][.,]\d{2}) to ([01][.,]\d{2})\)")

def num(s):
    return float(s.replace(",", "."))

def parse_between(lines):
    rows = []
    for ln in lines:
        m = ROW.match(ln)
        if m:
            rows.append(dict(author=m.group(1).strip(), ref=int(m.group(2)),
                             p=num(m.group(3)), lo=num(m.group(4)), hi=num(m.group(5))))
    return rows

def fig_rows(fig_no):
    txt = open(TXT, encoding="utf-8").read().split("\n")
    cap = next(i for i, l in enumerate(txt) if f"FIGURE {fig_no} " in l)
    pooled = max(i for i in range(cap - 40, cap) if "Pooled" in txt[i])
    hdr = max(i for i in range(pooled - 80, pooled) if "(95% CI)" in txt[i])
    return parse_between(txt[hdr + 1:pooled])

def main():
    se_rows = fig_rows(69)
    sp_rows = fig_rows(70)
    print(f"fig69 {len(se_rows)} rows, fig70 {len(sp_rows)} rows")
    assert len(se_rows) == 42 and len(sp_rows) == 42
    if [r["ref"] for r in se_rows] != [r["ref"] for r in sp_rows]:
        print("WARNING: order differs; matching by position anyway")
    years = load_years()
    rows_min, rows_max, problems = [], [], []
    for i, (se, sp) in enumerate(zip(se_rows, sp_rows)):
        s_se = solutions(se["p"], se["lo"], se["hi"])
        s_sp = solutions(sp["p"], sp["lo"], sp["hi"])
        flag = ""
        if not s_se or not s_sp:
            problems.append((se["author"], se["ref"], se if not s_se else sp, len(s_se), len(s_sp)))
            continue
        y = years.get(se["ref"], [None])[0]
        pairs = [(a, b) for a in s_se for b in s_sp
                 if 0.02 <= a[1] / (a[1] + b[1]) <= 0.95]
        if not pairs:
            pairs = [(a, b) for a in s_se for b in s_sp]
        pmin = min(pairs, key=lambda p: p[0][1] + p[1][1])
        pmax = max(pairs, key=lambda p: p[0][1] + p[1][1])
        for pick, rows in ((pmin, rows_min), (pmax, rows_max)):
            (tp, n1), (tn, n0) = pick
            rows.append(dict(cohort_id=i + 1, author=se["author"], ref=se["ref"],
                             year_pub=y, TP=tp, FN=n1 - tp, TN=tn, FP=n0 - tn,
                             n1=n1, n0=n0, flag=flag))
    for a, r, row, ns, nsp in problems:
        print("NO SOLUTION:", a, r, row, ns, nsp)
    for name, rows in (("ipg_min.csv", rows_min), ("ipg_max.csv", rows_max)):
        d = pd.DataFrame(rows)
        pool_se = d.TP.sum() / d.n1.sum()
        pool_sp = d.TN.sum() / d.n0.sum()
        print(f"{name}: k={len(d)} Se={100*pool_se:.1f} (print 75.0) "
              f"Sp={100*pool_sp:.1f} (print 90.0) years {d.year_pub.min()}-{d.year_pub.max()}")
        d.to_csv(name, index=False)

if __name__ == "__main__":
    main()
