"""hsroc: small-study effects on the hierarchical summary ROC curve.

The size-adjusted bivariate binomial meta-regression read through the HSROC model: the shape
of the summary curve, the latent accuracy and latent threshold trends with study size, the
likelihood-ratio test of the accuracy trend with the threshold trend free, and the Deeks test
as the comparator.

Modules
-------
data        study-level 2 x 2 tables of the two re-analysed reviews; logits, effective sample
            size and the size covariate
glmm        the bivariate binomial likelihood (adaptive Gauss-Hermite quadrature) and the full fit
normal      normal-approximation (REML) meta-regression: the starting value of the likelihood fit
fitting     the HSROC reading: shape, latent trends, constrained fits, likelihood-ratio test,
            delta-method contrasts, summary AUC; the fast fit used in the simulation study
funnel      the Deeks test on the effective-sample-size scale
design      the data-generating mechanism of the simulation study
"""
from . import data, design, fitting, funnel, glmm, normal  # noqa: F401

__version__ = "1.0.0"
