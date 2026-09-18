"""Helpers for reconstructing 2 x 2 tables from the printed summaries of Goodacre et al., HTA 2006;10(15).

The monograph reports each cohort's sensitivity and specificity as a proportion rounded to two decimals with an exact
(Clopper-Pearson) 95% interval, but not the group sizes.  ``solutions`` enumerates the integer pairs (x, n) whose
proportion and interval both reproduce a printed triple (p, lower, upper); ``load_years`` reads publication years from
the monograph's reference list (the ``pdftotext -raw`` conversion, in which entries are contiguous)."""
import re
import numpy as np
from scipy.stats import beta


def cp_ci(x, n, a=0.05):
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


TOL = 0.005 + 1e-9   # printed values are rounded to 2 dp (half-up)


def solutions(phat, lo, hi, nmax=4000):
    """All (x, n) whose phat and exact CI round to the printed values."""
    sols = []
    for n in range(1, nmax + 1):
        base = int(round(phat * n))
        for x in {base - 1, base, base + 1,
                  int(np.floor(phat * n)), int(np.ceil(phat * n))}:
            if not (0 <= x <= n):
                continue
            if abs(x / n - phat) > TOL:
                continue
            l, h = cp_ci(x, n)
            if abs(l - lo) <= TOL and abs(h - hi) <= TOL:
                sols.append((x, n))
    return sorted(set(sols), key=lambda s: s[1])


def load_years(raw="hta1015_raw.txt"):
    """Parse references from column-order text: entries are contiguous there."""
    txt = open(raw, encoding="utf-8").read().replace("\x0c", "\n")
    marks = [(int(m.group(1)), m.start(), m.end())
             for m in re.finditer(r"(?m)^(\d{1,3})\.\s", txt)]
    years = {}
    for i, (num, st, en) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else st + 600
        chunk = txt[en:end][:600]
        m = re.search(r"(19[5-9]\d|200[0-5])\s*[;,]", chunk)
        if not m:
            m = re.search(r"\b(19[5-9]\d|200[0-5])\b", chunk)
        if m:
            years.setdefault(num, []).append(int(m.group(1)))
    return {k: v for k, v in years.items()}
