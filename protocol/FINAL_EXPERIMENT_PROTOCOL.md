# Final experiment protocol — Stage 2D-E

Status: frozen before Stage 2D-E derived-result generation.

## Immutable baseline

- Formal execution commit: `d745e5412c1d530a2ae64e2eaa42c85c1f64e419`
- Baseline delivery commit: `a31c4eda5be8830ea06f6229044d853e5064c112`
- Schema: `stage2_v2`
- Formal baseline: `results/stage2/`
- Derived destination: `results/stage2de/`

The baseline is byte-preserved. Stage 2D-E does not train a new model unless a later written rerun
justification identifies a contract-required field that cannot be derived.

## Final task and source-family registry

| Dataset | Source family | Entity unit | Analysis role | Complete budgets |
|---|---|---|---|---|
| `acsincome_region` | `acs` | person | common-five | 0,25,50,100,250,500 |
| `acsfoodstamps_household` | `acs` | household | common-five | 0,25,50,100,250,500 |
| `brfss_diabetes` | `brfss` | respondent | common-five | 0,25,50,100,250,500 |
| `nhanes_lead` | `nhanes` | respondent | common-five rare-positive stress | 0,25,50,100,250,500 |
| `diabetes_readmission_patient_index` | `uci_diabetes` | patient | common-five | 0,25,50,100,250,500 |
| `college_scorecard_2018` | `college_scorecard` | institution | small-domain stress | 0,25,50,100 |

College Scorecard budgets 250 and 500 remain explicit `not_available` results and are not inserted
into the common-five curve.

## Frozen data roles and domains

Seven entity-disjoint roles are retained: `source_train`, `source_tune`,
`source_probability_calibration`, `source_conformal_calibration`, `source_id_test`,
`target_adaptation_pool`, and `target_final_test`. Target acquisition is label-blind and nested;
the final target test is evaluation-only. Task-specific source/target domain definitions and entity
rules are uniquely recorded in `data/FINAL_DATASET_REGISTRY.csv` and
`data/FINAL_ENTITY_UNIT_AUDIT.csv`.

## Frozen model and sampling grid

- Models: `logistic_regression` with model seed 0; `xgboost_cpu` with model seeds 0 and 1.
- Split seeds: 0, 1 and 2.
- Acquisition: `blind_random` only.
- Positive-budget adaptation seeds: 0 through 99.
- Budget-zero sentinel adaptation seed: -1.
- Budgets: 0, 25, 50, 100, 250 and 500.
- All methods within a run use the same target `sample_hash`.

## Frozen methods

Probability rows contain `uncalibrated`, `intercept_only_mle`, `sigmoid`, `isotonic`, and
`jeffreys_smoothed_intercept_matching`. Uncalibrated predictions are stored at budget zero only.
Every adapted probability method uses its own source-fitted budget-zero baseline.

Conformal methods are `standard` and `mondrian`, nominal coverage 0.90, with score sources
`base_uncalibrated_probability` and supplementary `source_sigmoid_probability`.

## Final technical scope

- H1: predicted support versus empirical support, algorithm estimability and finite thresholds.
- H2: paired complexity-budget benefits and harms, positive values meaning improvement.
- H3: common-five complete curves, separate College stress analysis, task/family equal weighting,
  LOTO and LOFO.
- H4: source-ID to target-final discrimination/reliability changes without causal interpretation.
- H5: direct Mondrian-minus-Standard comparison on identical samples.
- H6: `not_in_final_scope`; no new sampling experiment.
- Simulations: existing formal mechanistic grids only, kept separate from real tasks.

## Superseded protocol records

Earlier candidate and stage-specific protocols remain immutable historical evidence. For final
Stage 2D-E experiment interpretation they are `SUPERSEDED`; this document and
`protocol/FINAL_ANALYSIS_PLAN.md` are the unique active technical records. Historical files are not
deleted or rewritten.
