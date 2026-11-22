# Results summary — interpretable triage paper (JLA submission)

All numbers below are produced by the scripts in this package (`analysis_core.py` + `run_*.py`).
**Spec:** eligible = still enrolled at checkpoint; label = at-risk (Fail/Withdraw) at course end;
8 features; preprocessing fit on development only (no leakage); modules BBB + DDD; 2013J = development,
2014J = validation; checkpoints Day 28/56/84. This summary reflects the revised (reviewer-response)
manuscript; it is a reference for verifying reproduction, not the authoritative result (the manuscript
and supplement are).

---

## Sample (`table1_sample.csv`)
Eligible N 1,372–1,835 per cohort/checkpoint; base rate 0.33–0.50; 15–25% of each module cohort had
already withdrawn before the checkpoint and are excluded from the eligible evaluation.

## Eligibility inflation — the central result (`run_mc1_inflation.py`)
The same LR model evaluated on the conventional cohort (already-withdrawn students included) versus the
eligible population:
- **Precision@20 inflates to a near-perfect 0.98–1.00** under the conventional cohort, versus the true
  **0.75–0.94** on eligible students.
- **AUC inflates by 0.05–0.09** (e.g., BBB Day 56: 0.86 vs 0.78).
- Prevalence rises 0.08–0.13; recall is roughly stable (its denominator grows in step).
Full per-checkpoint table with F1: supplement Table S2.

**Isolation check (`run_mc7_fixed_model_inflation.py`).** MC1 refits LR within each cohort, so its gap
could in principle mix an evaluation-population change with a training-population change. Holding the
fitted model fixed (one LR trained on the eligible population; only the evaluated set changes) reproduces
the inflation almost exactly — **mean +0.065 AUC, identical to the refit version, within 0.002 at every
checkpoint**. Eligibility inflation is therefore an artifact of *which students are evaluated*, not of
retraining on already-withdrawn students. Supplement Table S2b.

## Archetypes (`table2_archetypes.csv`; `fig_feature_distributions.png`)
Disengaged (≤1 active day): 3–11%, 72–90% at-risk · Engaged, no submission: 3–8%, 94–96% · Submitting,
low score (<40): <1–4%, 92–94% · On track (≥40): ~85%, 35–42%.

## Primary model — Logistic Regression (`table3_models.csv`; `fig_calibration.png`)
Plain (unweighted) LR; outcomes near-balanced so weighting unnecessary and it preserves calibration.
- **Dev CV AUC** 0.74–0.84 (rising with checkpoint).
- **2013J→2014J transfer AUC:** BBB 0.59 (D28) → 0.76 (D56) → 0.76 (D84); DDD 0.78 → 0.80 → 0.83.
- **BBB Day 28 does not generalize** across presentations (transfer AUC 0.59; tree 0.50); earliness is
  module-dependent (DDD usable from Day 28, BBB from Day 56).
- Coefficients led by mean_score (−1.6 to −1.8), then active_days, has_early_assessment.

## Calibration (`run_mc4_calibration.py`; `fig_calibration_before_after.png`)
Brier read against a no-skill reference (base-rate prediction = prev·(1−prev) ≈ 0.22–0.25).
- DDD near-ideal (slope ≈1.0, intercept ≈0, Brier 0.16–0.19, below reference).
- BBB over-confident out-of-cohort (slope ≈0.79, intercept ≈−1.2). **Correction is TWO-parameter
  (intercept AND slope) logistic recalibration — not intercept-only** — fitted on target-presentation
  outcomes; e.g., BBB Day 56 slope 0.79→0.99, intercept −1.25→0.00, Brier 0.240→0.166.
- **BBB Day 28 Brier (0.56) exceeds the reference (0.23): not deployable there.**

## Model comparison — CV and transfer (`run_mc3_transfer_bakeoff.py`, `compare_models_extended.py`)
- **CV (12 settings):** LR highest in all 12 (mean AUC 0.786); RF 0.766, GBM 0.755, unpruned tree 0.637.
- **Transfer:** LR highest mean (0.754) vs RF 0.749, GBM 0.710, tree 0.719; LR matches/exceeds the
  ensembles at every checkpoint except BBB Day 28 (where no model is deployable).
- Interpretability carries no accuracy cost under either evaluation.

## Leakage robustness (`run_mc2_leakage_check.py`)
Rebuilding assessment features from only submissions made by the checkpoint (strict availability)
changes eligible AUC by ≤0.006 (mean 0.003) — no material temporal leakage.

## Decision utility + earliness (`table5_utility.csv`; `fig_capture_burden.png`, `fig_earliness.png`)
Validation transfer, Day 56: top-10% flag catches ~29% (BBB) / ~22% (DDD) at **96–98% precision**
(BBB 98.3%, DDD 95.8%); top-20% catches ~44% / ~40% at 76% / 87%. Earliness: DDD from Day 28, BBB from Day 56.

## Longitudinal tracking (`fig_trajectories.png`)
Risk rankings highly stable (Spearman 0.87–0.99); tracking confirms and refines a stable ranking rather
than revealing many surprise collapses (partly by construction — features are cumulative).

## Visualization (`fig_pca_feature_view.png`; `compare_embeddings.py`)
PCA primary (perfectly reproducible, interpretable axes, PC1 50% + PC2 18% = 68% in 2D). Procrustes
disparity: PCA 0.00, UMAP 0.04, t-SNE 0.18 (t-SNE least reproducible and worst archetype separation).

## Fairness / equity audit with CIs (`run_mc5_fairness_ci.py`)
Validation transfer, Day 56, global 20% cutoff, 95% bootstrap CIs (2,000 resamples).
- Capture/precision similar across gender, age, disability **but CIs are wide and overlap** (e.g.,
  disability Y capture 0.40 [0.29, 0.53] vs N 0.45 [0.41, 0.49]) — the data establish neither equity
  nor disparity; smallest groups (disability Y; Unknown-IMD n=75, precision [0.33, 1.0]) most uncertain.
- Flagging rate tracks base rate (most-deprived IMD and disability=Y flagged more). A single calibrated
  threshold cannot also equalize flag rates when base rates differ. Group-specific thresholds change the
  decision rule, not the probability calibration — a transparent policy option.
- IMD direction: lower band = more deprived.

## Outcome-definition sensitivity (`run_mc6_outcome_sensitivity.py`)
Withdrawal is 24–48% of the eligible at-risk group. Same features separate each mode from success:
Fail-vs-pass AUC 0.75–0.83; withdraw-vs-pass 0.69–0.84 (withdrawal less well, especially BBB) — supports
the combined outcome while treating the triage as prioritizing attention, not diagnosing cause.

## Similarity-network ablation (`repro2.py`)
On the eligible population LR matches/beats the network artifact at every operating point; its
"near-perfect" disconnected groups were 77–86% already-withdrawn students and collapse once eligibility
is enforced. Graph structure earns no place over transparent models (documented negative result).

---

### Caveats (stated in the paper, not buried)
1. Day 28 prediction is module-dependent and weak for BBB across cohorts.
2. BBB probabilities need a two-parameter recalibration (intercept and slope) to the target presentation.
3. Longitudinal stability is partly a consequence of cumulative features.
4. A single calibrated threshold cannot equalize flag rates across groups with differing base rates.
5. Weaker performance for the missing-IMD subgroup.
6. At-risk combines failure and withdrawal; withdrawal is predicted less well (sensitivity analysis).
7. Within-institution temporal transfer only; cross-institution generalization not established.
