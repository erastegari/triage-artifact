# Interpretable Triage for Early Student Support — Reproducibility Package

This repository reproduces every table and figure in the paper *Interpretable Triage for Early
Student Support: A Transparent, Equity-Aware Early-Warning Approach Evaluated on Still-Enrolled
Students* from the public Open University Learning Analytics Dataset (OULAD).

All results follow a single frozen specification (`analysis_core.py`): the eligible population is
students still enrolled at each checkpoint; the label is at-risk (Fail/Withdrawn) at course end;
eight features are used; and all preprocessing is fit on the development presentation only.

## 1. Requirements

- Python 3.11+
- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## 2. Get the data

This package does **not** redistribute OULAD. Download it (CC BY 4.0) from the Open University:
<https://analyse.kmi.open.ac.uk/open_dataset>

Unzip it and place the CSV files directly in the `data/` folder (or point `OULAD_DATA` at them).
The scripts use: `studentInfo.csv`, `studentRegistration.csv`, `studentAssessment.csv`,
`assessments.csv`, `vle.csv`, `courses.csv`, and `studentVle.csv`.

```
data/
  studentInfo.csv
  studentRegistration.csv
  studentAssessment.csv
  assessments.csv
  vle.csv
  courses.csv
  studentVle.csv        <- large raw interaction log
```

## 3. Preprocess (once)

`studentVle.csv` is very large; this step filters it to the two modules used in the study and
writes `data/vle_bbb_ddd.csv`:

```bash
python prepare_vle.py
```

## 4. Run everything

```bash
python run_all.py
```

Outputs (tables as CSV, figures as PNG, tree rules as text) are written to `./results/` and the
package root. To use a data folder elsewhere:

```bash
OULAD_DATA=/path/to/oulad python run_all.py
```

## 5. What each script produces

| Script | Produces |
|---|---|
| `run_foundation.py` | Table 1 (sample), Table 2 (archetypes), feature-distribution figure |
| `run_models.py` | Table 3 (LR + shallow tree; dev-CV and 2013J→2014J transfer; calibration), LR coefficients, tree rules, calibration figure |
| `run_utility.py` | Decision-utility table (capture/precision at 10/20/30% budgets), earliness and capture–burden figures |
| `run_longitudinal.py` | Risk-trajectory figure, ranking-stability and tier-transition stats, PCA + feature-view figure |
| `run_fairness.py` | Equity audit table (gender, age, SES/IMD, disability) |
| `compare_models_extended.py` | Model bake-off: LR vs shallow tree vs unpruned tree vs random forest vs gradient boosting |
| `compare_predictors.py` | LR vs shallow tree detail, AUC-by-checkpoint figure |
| `compare_embeddings.py` | PCA vs t-SNE vs UMAP (reproducibility + archetype separation) |
| `repro2.py` | Similarity-network ablation (graph structure adds no value over the features) |
| `run_mc1_inflation.py` | Conventional vs eligible evaluation — quantifies eligibility inflation (AUC, precision, recall, F1) |
| `run_mc2_leakage_check.py` | Temporal-leakage robustness: strict submission-date vs due-date assessment features |
| `run_mc3_transfer_bakeoff.py` | LR / tree / random forest / gradient boosting under cross-presentation transfer |
| `run_mc4_calibration.py` | Reference Brier + two-parameter recalibration, before/after (table + figure) |
| `run_mc5_fairness_ci.py` | Subgroup equity audit with 95% bootstrap confidence intervals |
| `run_mc6_outcome_sensitivity.py` | Fail-vs-Withdraw outcome-definition sensitivity |

## 6. Expected key results (for verification)

- **Eligible sample (Table 1):** e.g. BBB-2013J Day 28 → 1,835 eligible, base rate 0.416; 15–25% of
  each cohort already withdrew before a checkpoint and is excluded.
- **Primary model (Table 3):** logistic-regression transfer AUC ≈ 0.76 (BBB, Day 56) and ≈ 0.80
  (DDD, Day 56); BBB Day 28 transfer AUC ≈ 0.59 (does not generalize early).
- **Model comparison:** logistic regression has the highest AUC in all 12 module×checkpoint settings;
  random forest and gradient boosting do not beat it.
- **Decision utility:** at Day 56, the top-10% risk flag recovers ~29% (BBB)/~22% (DDD) of at-risk
  students at 97–98% precision.
- **Embeddings:** PCA Procrustes disparity 0.00 (perfectly reproducible); t-SNE 0.18 (least).

Full expected values are documented in `docs/RESULTS_SUMMARY.md`; the model-selection rationale is in
`docs/METHOD_CHOICE_COMPARISON.md`.

## 7. Notes on reproducibility

- Random seeds are fixed (`random_state=42`) throughout; cross-validation folds are seeded.
- t-SNE and UMAP layouts vary across versions/platforms by design (this is itself a finding — see
  `compare_embeddings.py`); PCA and the predictive results are deterministic given the data.
- A one-parameter calibration drift for the BBB module across presentations is expected and discussed
  in the paper.

## 8. Citation and license

- **Data:** Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset.
  *Scientific Data, 4*, 170171. https://doi.org/10.1038/sdata.2017.171 (OULAD is licensed CC BY 4.0.)
- **Code:** released under the MIT License (see `LICENSE`).
- Please cite the paper when using this code.
