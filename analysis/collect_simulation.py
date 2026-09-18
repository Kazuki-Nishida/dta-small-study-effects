#!/usr/bin/env python3
"""Collect the per-setting summaries of the simulation (results/simulation/<cell>.json) into one table, one row per
setting and procedure: the setting (block, k, rho, lambda, rho_s, delta, seed), the design quantities (sigma_alpha,
sigma_theta, delta/sigma_alpha, the true lnDOR slope b_DOR per SD of size), the numbers of generated, valid and
rejecting replicates, the rejection rate (valid denominator) with its Monte-Carlo standard error, the rate with all
generated replicates as denominator, and the diagnostics (non-convergence of the full and the constrained fit, boundary
hits, invalid p values, negative likelihood ratios, failed replicates).  Writes results/simulation_summary.csv and prints
the completion status against analysis/simulation_settings.csv.
Usage: python analysis/collect_simulation.py"""
import os, glob, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, ".."))
SIM = os.path.join(ROOT, "results", "simulation")


def main():
    man = pd.read_csv(os.path.join(HERE, "simulation_settings.csv"), dtype={"lam_tag": str})
    order = {c: i for i, c in enumerate(man["cell"])}
    rows = []
    for f in sorted(glob.glob(os.path.join(SIM, "*.json")), key=lambda p: order.get(os.path.basename(p)[:-5], 10 ** 6)):
        c = json.load(open(f))
        base = dict(cell=c["cell"], block=c.get("block"), k=c["k"], rho=c["rho"], lam=c["lam"], lam_tag=c["lam_tag"],
                    rho_s=c["rho_s"], delta=c["delta"], seed=c["seed"], reps=c["reps"], sigma_alpha=c["sigma_alpha"], sigma_theta=c["sigma_theta"],
                    delta_over_sigma_alpha=c["delta_over_sigma_alpha"], b_dor=c["b_dor"], mean_s_sd=c["mean_s_sd"],
                    nonconverged_full=c["nonconverged_full"], nonconverged_null=c["nonconverged_null"], boundary_full=c["boundary_full"], boundary_t=c["boundary_t"],
                    boundary_sig=c["boundary_sig"], boundary_null=c["boundary_null"], hessian_not_invertible=c["hessian_not_invertible"], invalid_any=c["invalid_any"],
                    negative_lr=c["negative_lr"], min_lr_raw=c["min_lr_raw"], failed_replicates=c["failed_replicates"], mean_lam_hat=c["mean_lam_hat"],
                    numpy=c.get("numpy"), scipy=c.get("scipy"), seconds=c.get("seconds"))
        for m, r in c["methods"].items():
            rows.append(dict(base, method=m, n_generated=r["n_generated"], n_valid=r["n_valid"], n_invalid=r["n_invalid"], n_reject=r["n_reject"],
                             rate=r["rate"], mcse=r["mcse"], rate_all_denominator=r["rate_all_denominator"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "simulation_summary.csv"), index=False)
    done = set(df["cell"]) if len(df) else set()
    missing = man[~man["cell"].isin(done)]
    print(f"{len(done)} of {len(man)} settings summarised; missing {len(missing)}: " + ", ".join(missing["cell"].tolist()[:12]) + (" ..." if len(missing) > 12 else ""))
    if len(df):
        d = df.drop_duplicates("cell")
        print("diagnostics over completed settings: nonconverged_full max", d["nonconverged_full"].max(), "| boundary_full max", d["boundary_full"].max(),
              "| invalid_any max", d["invalid_any"].max(), "| negative_lr max", d["negative_lr"].max(), "| failed max", d["failed_replicates"].max())


if __name__ == "__main__":
    main()
