# results/ — the numbers behind the paper

All files are written by the scripts in `analysis/`. Every random stream is
`numpy.random.default_rng` with the seed recorded in `analysis/simulation_settings.csv` and in
the result files, so every number is reproduced exactly by re-running. Standard errors use the
central-difference Hessian step recorded in `README.md` (1e-3 for the reviews, 1e-4 for the simulation
study and the parametric check).

| File | Script | Used for |
|---|---|---|
| `applications.json` | `analysis/applications.py` | Table 2 and Figure 3 (`scripts/fig_apps.py`). One entry per review: `FIT` (published counts) and `IPG` (minimum-count reconstruction at the top level, maximum-count endpoint under `max_endpoint`). For each fit: the shape `lam` with its interval, `gamma_alpha` and `gamma_theta` (the latent trends: est, se, p, ci), `lrt_alpha` and `lrt_theta` (likelihood-ratio tests: stat, p from chi^2_1, p_t from t_{k-2}), `betaA_1` (the lnDOR trend of the binomial fit, Table 2, second row), `betaA_H` (the accuracy-coordinate contrast, the fitted line of Figure 3), `lnDOR_threshold_part` and `lnDOR_accuracy_part` (the decomposition of the model's lnDOR trend), `deeks_slope`, `deeks_se`, `p_deeks`, and the seven fitted parameters. |
| `applications_fit_loo.json` | `analysis/applications_loo.py FIT` | supplement Section D.1: the 23 leave-one-out refits of the FIT review with the ranges quoted in the text |
| `fitted_parameters.json` | `analysis/fitted_parameters.py` | supplement Table S9: the seven fitted parameters, the latent residual standard deviations, bound flags, the tests, and the observed ranges of sensitivity and false-positive rate, for FIT and both IPG endpoints |
| `fit_level_checks.json`, `fit_level_checks_acc_null.csv` | `analysis/fit_level_checks.py` | supplement Section D.1 (parametric check): 2000 replicates in the design of the FIT review generated from the accuracy-null constrained fit (`generating_points.acc_null`), all seven parameters re-estimated in every replicate; the JSON holds the generating point and the rejection rates with Monte-Carlo standard errors, the CSV one row per replicate |
| `simulation/<cell>.json`, `simulation/<cell>.npz` | `analysis/simulate.py` | the 80 simulation settings of supplement Section C (Figures 1 and 2 of the main text; Tables S1 to S7), 1000 replicates each. The npz file holds every replicate's counts (`TP`, `FN`, `FP`, `TN`, one row per replicate), `s_sd`, the log-likelihoods of the full and the constrained fit, convergence and boundary flags, the raw likelihood ratio (`lr_raw`, before clipping at 0), the fitted shape `lam_hat`, the Deeks slope, the two contrast estimates and the p values `p_deeks`, `p_c1`, `p_hsW`, `p_hsL`, `p_hsLt`. The JSON holds the setting, its design quantities (`sigma_alpha`, `sigma_theta`, `delta_over_sigma_alpha`, `b_dor`, the true lnDOR slope per SD of size), the per-procedure counts and rates (valid replicates as denominator, with Monte-Carlo SE; `rate_all_denominator` counts an invalid p value as no rejection), the diagnostics of Tables S4 to S7 and the library versions used |
| `simulation_summary.csv` | `analysis/collect_simulation.py` | one row per setting and procedure, gathered from the JSON summaries |

Procedures in the simulation files: `deeks` (the Deeks test), `c1` (the lnDOR trend of the
binomial fit) and `hsLt` (the proposed test, signed likelihood-ratio root referred to t_{k-2}) are
the three procedures of the paper; `hsW` (the Wald contrast of the accuracy coordinate) and `hsL`
(the likelihood-ratio statistic referred to chi^2_1) are the auxiliary test forms of Tables S4 to
S7. Settings are named `K<k>-r<rho>-l<lambda>-rs<rho_s>-d<delta>`; the four blocks of settings are
described in `README.md` and in `analysis/simulation_settings.csv`.
