"""Study-level data of the two re-analysed reviews.

Each file gives one 2 x 2 table per study (TP, FN, FP, TN).  See ``data/README.md`` for the
sources of both reviews and the reconstruction of the IPG tables from the published summaries.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

REVIEWS = {
    "FIT": ("fit_crc_refpos.csv", "Faecal immunochemical tests for colorectal cancer, screening programmes (reference standard: positive)"),
    "IPG": ("ipg_min.csv", "Impedance plethysmography for deep-vein thrombosis (minimum-count reconstruction)"),
}
ENDPOINTS = {
    "IPG_max": ("ipg_max.csv", "Impedance plethysmography for deep-vein thrombosis (maximum-count reconstruction)"),
}


def load_review(key):
    """Return TP, FN, FP, TN (int arrays) for a review key or a CSV path."""
    if key in REVIEWS:
        path = DATA_DIR / REVIEWS[key][0]
    elif key in ENDPOINTS:
        path = DATA_DIR / ENDPOINTS[key][0]
    else:
        path = Path(key)
    d = pd.read_csv(path)
    return tuple(d[c].values.astype(int) for c in ("TP", "FN", "FP", "TN"))


def logits(TP, FN, FP, TN):
    """Continuity-corrected empirical logits of sensitivity and of the false-positive rate."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    return np.log((TP + .5) / (FN + .5)), np.log((FP + .5) / (TN + .5))


def ess(TP, FN, FP, TN):
    """Effective sample size ESS_i = 4 n1_i n0_i / (n1_i + n0_i)."""
    TP, FN, FP, TN = (np.asarray(v, float) for v in (TP, FN, FP, TN))
    n1 = TP + FN; n0 = FP + TN
    return 4 * n1 * n0 / (n1 + n0)


def size_covariate(TP, FN, FP, TN):
    """s_i = 1/sqrt(ESS_i) and the centred covariate x_i = s_i - s_bar."""
    s = 1 / np.sqrt(ess(TP, FN, FP, TN))
    return s, s - s.mean()
