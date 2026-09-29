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
    """All (x, n), n <= nmax, whose proportion and exact interval round to the printed values: every
    integer x in the band |x/n - phat| <= TOL is examined (for large n several counts share one printed
    proportion), and the pairs are returned ordered by group size n, then by count x."""
    n = np.arange(1, nmax + 1)
    xlo = np.maximum(np.ceil(n * (phat - TOL) - 1e-12).astype(int), 0)
    xhi = np.minimum(np.floor(n * (phat + TOL) + 1e-12).astype(int), n)
    keep = xhi >= xlo
    ns = np.repeat(n[keep], xhi[keep] - xlo[keep] + 1)
    xs = np.concatenate([np.arange(a, b + 1) for a, b in zip(xlo[keep], xhi[keep])]) if keep.any() else np.array([], int)
    ok = np.abs(xs / ns - phat) <= TOL
    xs, ns = xs[ok], ns[ok]
    with np.errstate(all="ignore"):
        l = np.where(xs == 0, 0.0, beta.ppf(0.025, np.maximum(xs, 1), ns - xs + 1))
        h = np.where(xs == ns, 1.0, beta.ppf(0.975, xs + 1, np.maximum(ns - xs, 1)))
    m = (np.abs(l - lo) <= TOL) & (np.abs(h - hi) <= TOL)
    return sorted(zip(xs[m].tolist(), ns[m].tolist()), key=lambda s: (s[1], s[0]))


KEEP = 12   # candidate (count, size) pairs kept per coordinate: ordered by group size, then by count (see README.md)
PREV_LO, PREV_HI = 0.02, 0.95   # prevalence band for pairing the sensitivity and specificity candidates


def pair_candidates(s_se, s_sp, keep=KEEP):
    """Pair the truncated candidate sets under the prevalence band (all pairs if none qualifies)."""
    s_se, s_sp = s_se[:keep], s_sp[:keep]
    pairs = [(a, b) for a in s_se for b in s_sp if PREV_LO <= a[1] / (a[1] + b[1]) <= PREV_HI]
    return pairs or [(a, b) for a in s_se for b in s_sp]


def endpoints(pairs):
    """Minimum-count and maximum-count pairs (total group size; ties by the sensitivity group size)."""
    key = lambda p: (p[0][1] + p[1][1], p[0][1])
    return min(pairs, key=key), max(pairs, key=key)


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
