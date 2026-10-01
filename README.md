# dta-small-study-effects

Code, data and results for

> Nonomiya Y, Nishida K. *Small-study effects on the hierarchical summary ROC curve: latent
> accuracy and threshold trends in meta-analysis of diagnostic test accuracy.* arXiv preprint
> arXiv:2609.38297 [stat.ME], 2026. https://arxiv.org/abs/2609.38297

In meta-analysis of diagnostic test accuracy, the Deeks test assesses small-study effects
through the log diagnostic odds ratio (lnDOR). Under the hierarchical summary ROC (HSROC) model
each study has a latent threshold and a latent accuracy, and the lnDOR moves with both when the
summary curve is asymmetric. The paper defines the target of an accuracy-specific assessment as
the size trend in latent accuracy, with the latent threshold free to vary with study size, and
shows that the size trend in the true lnDOR is a shape-dependent mixture of the two latent trends
whose weight on the threshold trend vanishes only for a symmetric curve, so that the null
hypothesis of the Deeks test coincides with the accuracy null only then. The accuracy null is the
hypothesis that studies of all sizes share one summary curve. The proposed procedure is a
likelihood-ratio test of it within the size-adjusted bivariate binomial meta-regression read as
an HSROC model, with the threshold trend left free. Two things change between the Deeks test and
the proposed test, the estimator (weighted least squares on continuity-corrected logits versus
the binomial likelihood) and the axis (lnDOR versus latent accuracy); the lnDOR trend of the same
binomial fit is carried alongside as the control that separates the two. This repository contains
the fitting code, the simulation study, the two re-analysed reviews and the scripts that
regenerate every table and figure of the paper and its supplement.

## Contents

```
hsroc/                  the package
  glmm.py               size-adjusted bivariate binomial meta-regression (adaptive Gauss-Hermite quadrature)
  fitting.py            the HSROC reading: shape lambda, latent accuracy and threshold trends, constrained
                        fits, likelihood-ratio test, delta-method contrasts, summary AUC; the fast fit
  funnel.py             the Deeks test on the effective-sample-size scale
  normal.py             normal-approximation (REML) meta-regression: the starting value of the likelihood fit
  design.py             the data-generating mechanism of the simulation study
  data.py               loading of the two reviews; logits, ESS, size covariate
analysis/               the simulation study (simulate.py with its manifest simulation_settings.csv and
                        collect_simulation.py), the applications (applications.py, the leave-one-out fits
                        applications_loo.py, the fitted-parameter table fitted_parameters.py) and the
                        parametric check in the design of the FIT review (fit_level_checks.py)
data/                   study-level 2 x 2 tables of the two reviews (FIT: published counts; IPG: reconstructed)
                        and the reconstruction scripts (data/README.md)
results/                the result files behind every number of the paper and the supplement (results/README.md)
figures/                Figures 1 to 3
scripts/                one script per figure (fig_sim.py: Figures 1 and 2; fig_apps.py: Figure 3)
tests/                  pytest smoke and regression tests
```

## Installation

Python 3.10 or later with NumPy, SciPy, pandas and matplotlib.

```bash
git clone https://github.com/Kazuki-Nishida/dta-small-study-effects.git
cd dta-small-study-effects
pip install -e .          # or: pip install -r requirements.txt
pytest -q                 # under a minute; refits both reviews and re-analyses stored simulation replicates
```

The results in `results/` were produced with Python 3.11, NumPy 2.4.4 and SciPy 1.17.1 (33 of
the 80 simulation settings with NumPy 2.2.6 and SciPy 1.15.3; each setting's summary records the
versions used), single-threaded (`OPENBLAS_NUM_THREADS=1`, set by the scripts). All random
streams are `numpy.random.default_rng` with the seeds recorded in `analysis/simulation_settings.csv`
and in the result files, and the fits consume no random numbers, so every simulated data set is
regenerated exactly from its seed. With the recorded library versions the fitted values and p
values are reproduced exactly as well; with other versions the optimizer may stop at slightly
different points, so fitted values can differ beyond the fourth significant digit and an
occasional rejection decision near the 0.10 boundary can change.

Standard errors come from a central-difference Hessian of the marginal log-likelihood
(`hsroc/glmm.py`). The step is 1e-3 for the two reviews (`glmm.HESSIAN_STEP_APPLICATIONS`, used by
`analysis/applications.py`, `applications_loo.py` and `fitted_parameters.py`) and 1e-4 for the
simulation study and the parametric check (`glmm.HESSIAN_STEP_SIMULATION`, used by
`analysis/simulate.py` and `fit_level_checks.py`), the step with which the stored simulation results
were computed. In four simulation settings re-analysed from the stored replicates (4,000 replicates
in all), the two steps gave the same decision at the 0.10 level in every replicate. In the FIT
review, whose fit has a residual correlation of 0.98 and group sizes up to 7.5e5, a step of 1e-4 is
at the edge of the range in which numerical noise enters the second differences (the standard error
of the lnDOR trend of the binomial fit moves by 0.4%), whereas steps from 1e-3 to 1e-2 agree to four
significant digits; the review results therefore use 1e-3.

## Analysing a review

```python
import numpy as np
from hsroc import data, fitting, funnel, glmm, normal

TP, FN, FP, TN = data.load_review("FIT")                        # or "IPG", or a path to a CSV with TP, FN, FP, TN
s, x = data.size_covariate(TP, FN, FP, TN)                      # s = 1/sqrt(ESS), x = s - mean(s)
o = glmm.fit(TP, FN, FP, TN, x, start=glmm.start_from_normal(normal.analyse(TP, FN, FP, TN)["fit"]))

q = fitting.hsroc_quantities(o["theta"], o["V"], len(TP), x=x)
q["lam"], q["lam_ci"]                   # shape of the summary curve with its delta-method interval (1 = symmetric)
q["gamma_alpha"], q["gamma_theta"]      # latent accuracy and latent threshold trends (est, se, p, ci)
null = fitting.fit_null(TP, FN, FP, TN, x, o["theta"], "alpha")
fitting.lrt(o["nll"], null["nll"], k=len(TP))   # likelihood-ratio test of the accuracy null: stat, p (chi^2_1), p_t (t_{k-2})

eta, phi = data.logits(TP, FN, FP, TN)
funnel.deeks_test(eta - phi, data.ess(TP, FN, FP, TN))   # the Deeks test: slope, p
q["betaA_1"]                            # the lnDOR trend of the binomial fit (same axis as the Deeks test): est, se, p
```

## Reproducing the paper

| Output | Script | Result file | Runtime (one core) |
|---|---|---|---|
| Table 2, Figure 3 | `analysis/applications.py` | `results/applications.json` | seconds |
| Table S9 (fitted parameters of both reviews, both IPG endpoints) | `analysis/fitted_parameters.py` | `results/fitted_parameters.json` | seconds |
| Section D.1, leave-one-out fits of the FIT review | `analysis/applications_loo.py FIT` | `results/applications_fit_loo.json` | about a minute |
| Section D.1, parametric check in the design of the FIT review | `analysis/fit_level_checks.py` | `results/fit_level_checks.json`, `fit_level_checks_acc_null.csv` | about 50 minutes |
| Figures 1 and 2, Tables S1 to S7 (the 80 simulation settings) | `analysis/simulate.py --cells 0:80`, then `analysis/collect_simulation.py` | `results/simulation/<cell>.json` and `.npz`, `results/simulation_summary.csv` | 5 to 15 minutes per setting (about 15 CPU-hours; resumable) |
| Figures 1 to 3 | `scripts/fig_sim.py`, `scripts/fig_apps.py` | `figures/*.pdf`, `*.png` | seconds |

`make applications`, `make simulation`, `make collect`, `make fitcheck` and `make figures` run the
corresponding steps. `simulate.py` accepts `--cells a:b` or `--cells i,j,...` (indices into the
manifest) and `--reps` for quick checks, and `--chunk CELL --budget SECONDS` for time-budgeted
partial runs that are merged when complete.

## Simulation design

The generator is `hsroc/design.py` (supplement Section C.1): the latent threshold and the latent
accuracy of the HSROC model are independent normals with `sigma_eta * sigma_phi = 0.75^2` and
residual correlation `rho` of the logits; the shape `lambda = sigma_eta / sigma_phi` takes the
values 1/4, 1/2, 1/sqrt2, 1, sqrt2, 2 and 4; the mean operating point is (logit sensitivity 1.0,
logit false-positive rate -2.0); study sizes are log-normal (median 300, prevalence 0.35); a
threshold trend of strength `rho_s` residual SDs of the latent threshold per SD of `1/sqrt(ESS)`
and an accuracy trend of `delta` latent-accuracy units per SD; 1000 replicates per setting;
rejection at the 0.10 level. The 80 settings of `analysis/simulation_settings.csv` form four
blocks:

- `shape` (28): `k = 30`, `rho = 0.4`, the seven shapes under no trend, a threshold trend alone
  (`rho_s = 0.4`), an accuracy trend alone (`delta = 0.5`) and both trends (Figures 1(a), 2(a) and
  2(b); Tables S1 and S4);
- `threshold-strength` (7): `lambda = 1/2`, `rho_s` from 0.2 to 1.6 without an accuracy trend
  (with the two `shape` settings at `rho_s = 0` and 0.4: Figure 1(b); Tables S2 and S4);
- `k-comparison` (27): `k` in {10, 20, 50} at `rho = 0.4`, `lambda` in {1/2, 1, 2}, the three
  one-trend scenarios (Table S3; Tables S5 to S7);
- `rho-comparison` (18): `rho` in {0, 0.8} at `k = 30`, the same shapes and scenarios (Tables S3
  and S7). With `rho` the residual SDs of the latent variables change, so the accuracy trend of
  0.5 is 0.47, 0.61 and 1.05 residual SDs at `rho` = 0, 0.4 and 0.8.

## Results at a glance

Under a threshold trend alone, the false-alarm rate of the Deeks test for the accuracy question
rises with the asymmetry of the curve, from 0.136 at the symmetric curve to 0.357 and 0.311 at
`lambda` = 1/4 and 4, and the lnDOR trend of the binomial fit behaves the same way (0.099 at the
symmetric curve, 0.420 and 0.399 at the ends), so the false alarms come from the axis and not from
the estimator; the proposed likelihood-ratio test stays between 0.098 and 0.133 at every shape.
Under an accuracy trend alone the proposed test has higher power than the Deeks test at every
shape (0.785 to 0.866 against 0.529 to 0.702). At `lambda = 1/2` the two lnDOR-axis rates grow with
the strength of the threshold trend (Deeks 0.16 to 0.85, binomial fit 0.10 to 0.96 over 0 to 1.6)
while the proposed test stays near its level. Over the 51 settings with the accuracy null true the
proposed test lies between 0.085 and 0.134. In the two reviews (Table 2), the FIT screening review
(23 entries) has an asymmetric fitted curve (`lambda` = 2.11): the Deeks test (p = 0.41) and the
lnDOR trend of the binomial fit (p = 0.71) see nothing, while the proposed test finds a latent
accuracy trend (p = 0.029) accompanied by a threshold trend of the opposite sign on the lnDOR axis
(p = 0.081), the two cancelling in the lnDOR trend. Impedance plethysmography (42 cohorts) has a
nearly symmetric fitted curve (`lambda` = 1.00) and the three procedures agree on a small-study
effect (Deeks p = 0.012, binomial lnDOR trend p = 0.010, proposed p = 0.023).

## Data

`data/README.md` documents the two datasets: the 23 entries of the FIT screening review (exact
counts from the review's published data file, with a table of study characteristics) and the
42 cohorts of the IPG review, reconstructed from the published per-cohort estimates and exact
intervals of Goodacre et al. (2006) with the two extreme admissible reconstructions (minimum and
maximum count) carried through the analysis.

## License and citation

The code is released under the MIT License (see `LICENSE`). The data files reproduce published
study-level counts and are provided for reproducibility; please cite the original reviews
(Grobbee et al. 2022; Goodacre et al. 2006) when using them. To cite this repository, see
`CITATION.cff` (preprint: arXiv:2609.38297); the journal reference of the manuscript will be added
on publication.
