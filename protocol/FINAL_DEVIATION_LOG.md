# Final experiment deviation log — Stage 2D-E

## Frozen scope status

- H6: `final_scope_status = not_in_final_scope`.
- No new model, dataset, acquisition strategy or scientific question is permitted.
- Existing manuscript artifacts are `SUPERSEDED / NOT FOR SUBMISSION` and are not modified.

## Deviations and retained historical differences

### Historical formal-simulation scope retained

- Earlier candidate text mentioned eight mechanisms, three acquisitions, Jeffreys-smoothed
  probability simulation and 500 repetitions. The actually frozen and executed Stage 2 simulation
  evidence contains seven probability mechanisms, four conformal mechanisms, the recorded method
  grids, blind-random sampling and 200 repetitions per cell.
- Stage 2D-E is contractually limited to the existing formal mechanistic grids and does not add
  scenarios to reconcile superseded candidate text.
- Probability simulation historically treats sigmoid 5/5 support as a stability requirement;
  natural-task H1 uses the formal runner's actual 1/1 sigmoid estimability rule.

### Minimal probability-simulation replay authorized

- Existing simulation probability rows omit calibration intercept, calibration slope and their
  absolute errors. These quantities cannot be reconstructed from Brier, log loss and ECE.
- The exact existing grid is replayed with identical configuration and seeds under a new code
  commit, adding only the missing metrics. The entire replay file is used as one coherent
  Stage 2D-E simulation input; columns from old and new commits are not spliced.
- No real-task shard and no conformal simulation is rerun.

### Pre-derivation implementation correction

- An independent contract audit after the first analysis-code tag, but before H1-H5 derivation,
  found delivery/schema completeness gaps. The old tag is retained; the corrected implementation
  is frozen under `stage2de-analysis-code-frozen-v2`.
- The already passing probability replay remains bound to
  `7e32b9d37b0929ba5a408017dba839a016b1de86`; the derived-analysis run manifest records a separate
  v2 analysis commit. No replay output is column-spliced or relabelled.
- Corrections are technical only: exact required names/rates, figures/task counts, registry
  explicitness, result/freeze coverage and test/handoff verification. The scientific grid and
  estimands are unchanged.

### Pre-submission H5 estimand-label correction

- The original presentation label "all-attempt" was inaccurate for H5 metric differences. The
  archived code defines a metric contrast only when Standard and Mondrian are jointly successful
  and the contrast is finite; non-estimable attempts are not assigned zero effects.
- The primary conditional summary retains successful Mondrian outputs with infinite thresholds,
  takes the median within each task–model–budget cell, and then averages the 10 cell medians
  equally. The sensitivity summary additionally requires a finite Mondrian threshold and averages
  the available cell medians. Estimability and finite-threshold rates use available-attempt
  denominators.
- This correction changes labels and explanatory text only. It does not change the archived
  experiment grid, row selection, transformations, or numerical values.
