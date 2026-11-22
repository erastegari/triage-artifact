#!/usr/bin/env python3
"""Run the full analysis pipeline in order and write all tables/figures to ./results.
Assumes OULAD CSVs are in ./data (or OULAD_DATA) and that prepare_vle.py has been run.
"""
import os, subprocess, sys

SCRIPTS = [
    "run_foundation.py",          # Table 1 (sample), Table 2 (archetypes), feature distributions
    "run_models.py",              # Table 3 (LR + tree, transfer, calibration), coefficients, tree rules
    "run_utility.py",             # Table 5 (decision utility), earliness + capture-burden figures
    "run_longitudinal.py",        # trajectories, tier transitions; final PCA + feature view
    "run_fairness.py",            # Table 6 (equity audit)
    "compare_models_extended.py", # model bake-off (LR vs tree vs RF vs GBM)
    "compare_predictors.py",      # LR vs shallow tree detail + AUC-by-checkpoint figure
    "compare_embeddings.py",      # PCA vs t-SNE vs UMAP (reproducibility + archetype clarity)
    "repro2.py",                  # similarity-network ablation (network adds nothing)
    # Reviewer-response analyses (manuscript revision)
    "run_mc1_inflation.py",       # conventional vs eligible evaluation (eligibility inflation)
    "run_mc2_leakage_check.py",   # temporal-leakage robustness (strict submission-date features)
    "run_mc3_transfer_bakeoff.py",# LR/tree/RF/GBM under cross-presentation transfer
    "run_mc4_calibration.py",     # reference Brier + two-parameter recalibration (before/after)
    "run_mc5_fairness_ci.py",     # subgroup audit with bootstrap confidence intervals
    "run_mc6_outcome_sensitivity.py",  # Fail-vs-Withdraw outcome sensitivity
    "run_mc7_fixed_model_inflation.py",# inflation with the fitted model held fixed (isolates evaluation-cohort effect)
]

def main():
    vle = os.path.join(os.environ.get("OULAD_DATA") or os.path.join(os.path.dirname(__file__), "data"),
                       "vle_bbb_ddd.csv")
    if not os.path.exists(vle):
        print("Missing vle_bbb_ddd.csv — run `python prepare_vle.py` first.", file=sys.stderr)
        sys.exit(1)
    for s in SCRIPTS:
        print(f"\n{'='*70}\n# {s}\n{'='*70}")
        r = subprocess.run([sys.executable, s])
        if r.returncode != 0:
            print(f"ERROR in {s}", file=sys.stderr); sys.exit(r.returncode)
    print("\nAll scripts completed. Tables and figures are in ./results and the package root.")

if __name__ == "__main__":
    main()
