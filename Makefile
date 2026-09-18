# Reproduction targets.  Long-running targets print their progress; see README.md for runtimes.
PY ?= python3

.PHONY: test applications simulation collect fitcheck figures all

test:
	$(PY) -m pytest -q

applications:                      # Table 2, Figure 3 and the tables of Section D: results/applications.json, applications_fit_loo.json, fitted_parameters.json
	$(PY) analysis/applications.py
	$(PY) analysis/applications_loo.py FIT
	$(PY) analysis/fitted_parameters.py

simulation:                        # the 80 settings of Section C (1000 replicates each; about 15 CPU-hours; resumable): results/simulation/
	$(PY) analysis/simulate.py --cells 0:80

collect:                           # one row per setting and procedure: results/simulation_summary.csv
	$(PY) analysis/collect_simulation.py

fitcheck:                          # the parametric check in the design of the FIT review (2000 replicates, about 50 minutes): results/fit_level_checks.json
	$(PY) analysis/fit_level_checks.py

figures:                           # figures/*.pdf, *.png (Figures 1 and 2: fig_sim_null, fig_sim_power; Figure 3: fig_apps)
	$(PY) scripts/fig_sim.py
	$(PY) scripts/fig_apps.py

all: applications simulation collect fitcheck figures
