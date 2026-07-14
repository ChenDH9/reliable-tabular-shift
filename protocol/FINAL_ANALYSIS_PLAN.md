# Final analysis plan — Stage 2D-E

Status: frozen before derived-result generation.

## Global rules

- Read only the formal machine-readable results and frozen registries; do not read manuscript
  prose to select analyses.
- Preserve `success`, `not_available`, `not_estimable` and `failed` as explicit statuses.
- Treat task as the primary aggregation unit and source family as the secondary unit.
- Pair within task/split/base-model/model-seed/adaptation-seed/budget/sample-hash before aggregation.
- H2 and explicitly named `benefit_oriented_*` values use positive = improvement. Raw H4
  `delta_*` values are target minus source. Raw H5 `delta_*` values are Mondrian minus Standard;
  their direction is recorded per metric and is not silently reversed.
- Report descriptive uncertainty (median, IQR, p05, p95) and rates; do not run-level bootstrap.

## H1

For each task/model/method/budget, retain four distinct quantities:
`predicted_class_support_probability`, `empirical_class_support_rate`,
`empirical_algorithm_estimability_rate`, and `empirical_finite_threshold_rate`. Validate prediction
against empirical outcomes with MAE, RMSE, bias, median absolute error, p05/p95 residual and
`observed_minus_predicted`. Keep `true_composition_reference` separate from conservative
`prevalence_range_planning` values. Planning uses the task-independent, deployment-predeclared
binary-prevalence range [0.01, 0.99], never the target pool's true composition. For the symmetric
per-class support requirements used here, report the minimum exact finite-population probability
at the two attainable range boundaries and verify endpoint minimization against exhaustive small-
pool calculations in tests. The target pool composition appears only in retrospective
`true_composition_reference` rows.

Eligibility mapping is frozen as: intercept-only MLE and sigmoid >=1 of each class; isotonic n>=25,
>=5 per class and >=10 unique input scores; standard finite threshold is deterministic from total
calibration size and does not use class support; Mondrian uses the exact finite-threshold formula.

## H2

For `intercept_only_mle`, `jeffreys_smoothed_intercept_matching`, `sigmoid`, and `isotonic`, pair
positive-budget target rows with the same source-fitted method at budget zero. Produce run-, task-
and family-level summaries for log loss, relative log loss, Brier skill, absolute calibration
intercept and absolute calibration slope error. Positive values are benefit. Material thresholds
remain: absolute log-loss reduction >=0.002 and relative reduction >=1%; Brier-skill gain >=0.005.
Other metrics are descriptive and use zero as the sign boundary. Store availability and algorithm
estimability over all attempts separately from benefit/harm rates conditional on successful,
finite paired effects; never count `not_available` or `not_estimable` as no harm.

## H3

Build complete 0-500 curves only for the five tasks with every budget available. Analyze College
Scorecard separately at 0-100. Report absolute budget, fraction of target adaptation pool, observed
positive count and observed negative count. Collapse run effects within tasks first, then apply
equal task and equal source-family weighting. Recompute LOTO and LOFO from the paired H2 results.

## H4

Directly pair budget-zero `source_id_test` and `target_final_test` rows. The primary comparison uses
uncalibrated base-model probabilities; source-fitted calibration methods are supplementary. Every
raw `delta_*` is target-final minus source-ID, including Brier, log loss and absolute errors; an
explicit per-metric direction plus `benefit_oriented_change` is stored for cross-metric sign checks.
Pair conformal source/target rows separately by conformal method and score source to obtain
`delta_worst_class_coverage`, with base-uncalibrated score rows primary. Produce paired rows, task
summaries, continuous scatter plots, direction tables, task-level concordant/discordant counts,
and AUROC-change sensitivity tables using strict absolute changes `<0.01` and `<0.02`. Contract
names `delta_brier_skill`, `delta_log_loss_skill`, and `delta_calibration_slope_error` are canonical;
legacy `*_score`/`abs_*` spellings are compatibility views only.

## H5

Pair Mondrian and Standard rows on the exact dataset, split, base model, model seed, adaptation
seed, budget, score source, evaluation split and sample hash. Every contract `delta_*` is the raw
`Mondrian - Standard` difference, including worst-class undercoverage and set size. Additional
benefit-oriented columns may reverse lower-is-better metrics but never replace the required raw
delta. `delta_estimable_rate` and `delta_finite_threshold_rate` use pairs where both methods are
available and deliberately do not require joint metric success. Include common-five and rare-task
flags; keep `not_available` separate from `not_estimable`.

## Simulation summaries

Summarize the existing conformal grid and the contract-authorized deterministic probability-grid
replay by `shift_mechanism` (with `mechanism` retained as an identical compatibility alias), target
prevalence, budget and method. Compute estimability, harm/benefit relative to
the same mechanism/prevalence/budget/repetition uncalibrated row, log loss, calibration
intercept/slope errors, coverage, finite thresholds and set size. Do not merge simulation rows with
real-task rows and do not label simulations external validation. The historical simulation sigmoid
5/5 support gate is a simulation stability convention and is not substituted for the natural-task
runner's 1/1 algorithm-estimability rule.
