---
title: "Supplementary Material: Complexity and Label-Budget Trade-offs in Probability Updating and Class-Conditional Conformal Prediction under Natural Tabular Shifts"
bibliography: manuscript/writing_code_stage3q/references_stage3q.bib
link-citations: true
---

# Supplementary Material

## S1. Purpose and archived provenance

<!-- SID:S-S1-01 -->
This supplement provides additional methodological, diagnostic, and reproducibility details for
the analyses reported in the main manuscript. The natural-shift experiments correspond to
software commit `d745e5412c1d530a2ae64e2eaa42c85c1f64e419`; the archived experimental outputs and
analysis code correspond to commits `74f56923907b75cac638b411b5af6293e4f3a11e` and
`9836d7239f7c96d50e6f510d12fc1e8344752b9c`. All tables are rounded displays of archived
machine-readable records, whose source files are listed in `manuscript/source_data_v2/`.
Subsequent reproduction checks were conducted separately and did not replace or modify the
archived outputs used in the manuscript.

<!-- SID:S-S1-02 -->
The archived execution contained 54 completed model configurations and no technical failure row. The six
tasks, five source families, two base-model types, three split seeds, and frozen model-seed strata
are described in the main paper. Positive-budget target samples used 100 adaptation seeds per
configuration. The analyses reported in the manuscript use the archived outputs. A subsequent
representative ACS reproduction check was run separately and did not replace or modify those
outputs. An earlier probability-simulation replay added missing calibration metrics before the
archived analysis version and was not repeated for manuscript preparation.

## S2. Entity, domain, and role controls

<!-- SID:S-S2-01 -->
The entity audit reported equality of row count and unique-entity count for every final task table,
with zero duplicated entities. The entity units were person, household, respondent, respondent,
patient, and institution for ACS Income, ACS Food Stamps, BRFSS, NHANES, UCI diabetes, and College
Scorecard, respectively. UCI encounters were reduced to a deterministic patient index before role
assignment. Role hashes were generated after entity-level assignment, and no role cap used labels.

<!-- SID:S-S2-02 -->
Seven roles were mutually exclusive for each split: source training, source tuning, source
probability calibration, source conformal calibration, source in-domain testing, target adaptation,
and target final testing. The separate source calibration roles prevent probability fitting from
reusing the scores that define source conformal thresholds. Target final testing was never used for
model selection, acquisition, calibration, or threshold fitting. For a given adaptation seed,
positive budgets were nested prefixes of one label-blind deterministic ordering, and all methods
shared the resulting sample hash.

### S2.1 Exact task definitions and source versions

<!-- SID:S-S2-03 -->
ACS Income used 2018 ACS 1-year person PUMS records and the Folktables adult filter
`AGEP > 16`, `PINCP > 100`, `WKHP > 0`, and `PWGTP ≥ 1`; the project label was
`PINCP ≥ 56,000`. `DIVISION = 1` (New England) was target and other divisions were source
[@uscensus2018acspums; @uscensus2018pumsdict; @ding2021folktables]. ACS Food Stamps joined the
person and housing PUMS tables by household serial, then retained `RELP = 0`, `18 ≤ AGEP < 62`,
`PINCP ≤ 30,000`, and `HUPAC ∈ {1,2,3,4}`. `FS = 1` was positive; `DIVISION = 6` (East South
Central) was target. `PINCP` is personal income. Census `HUPAC` code 4 means no children, so this
cohort is not restricted to households with children.

<!-- SID:S-S2-04 -->
BRFSS used the 2015, 2017, 2019, and 2021 public-use files. `DIABETE3` or `DIABETE4` code 1 was
positive; valid codes 2, 3, and 4 were negative, while 7, 9, and missing values were excluded.
`_PRACE1 = 1` was source and codes 2–6 were target; this is a race-code partition and not
a White-non-Hispanic definition [@cdcbrfss2015; @cdcbrfss2017; @cdcbrfss2019; @cdcbrfss2021].
NHANES merged cycle-qualified `SEQN` identifiers across 1999–2000 through 2017–2018, retained
complete `LBXBPB`, `INDFMPIR`, and `RIDAGEYR` at age 18 or younger, labeled `LBXBPB ≥ 3.5` μg/dL,
and split target at `INDFMPIR ≤ 1.3` [@nchsnhanes1999_2018; @nchsnhanes2017demo;
@nchsnhanes2017lead]. The cutpoint is a benchmark informed by the CDC blood lead reference value,
not a toxicity or diagnosis threshold, and no survey weights were used [@ruckart2021blrv].

<!-- SID:S-S2-05 -->
The UCI task used the public 1999–2008 diabetes encounters, with a stable-hash patient index
encounter rather than a chronology claim. `admission_source_id = 7` (emergency room) was target,
and `readmitted != NO` was positive, including both `<30` and `>30` categories
[@clore2014diabetes130; @strack2014hba1c]. College used `MERGED2018_19_PP.csv`; `CONTROL = 1` or 2
was source, `CONTROL = 3` was target, and `C150_4 ≤ 0.50` was positive. `C150_4` is the applicable
IPEDS first-time, full-time cohort completion measure within 150% of normal time, not an all-student
pooled completion rate [@usedcollegescorecard2026; @usedcollegescorecarddict2026;
@usedcollegescorecarddocs2025].

## S3. H1 rate-level feasibility audit

<!-- SID:S-S3-01 -->
The H1 validation concerns event rates, not the discrepancy between a single binary realization and
its probability. The frozen run-level values called MAE and RMSE equal functions of $|Y-p|$ for a
single support event and remain nonzero under a correctly specified Bernoulli probability. They are
therefore not used here as probability-versus-rate calibration measures. The corrected analysis
groups the 100 adaptation draws within each design stratum, compares empirical event rate with its
finite-population probability, and then applies task- and family-equal summaries. Model-stratified
rows that share one label sample are design duplicates rather than independent repetitions.

<!-- SID:S-S3-02 -->
Across 1,260 model-stratified groups, the task-equal absolute class-support rate gap was 0.002433,
the family-equal gap 0.002658, and the task-equal signed gap +0.000106. Wilson intervals covered the
predicted rate in 100% of groups. Algorithm eligibility equaled the method-specific class-support
indicator in all 126,000 method–model-stratified available records after score-source
deduplication: the observed algorithm-minus-support
difference had minimum and maximum zero. This is a technical validation of the archived implementation and
hypergeometric support calculation, not a separate theory result. Classwise finite-threshold
requirements and sparse-class limitations are established in conformal literature
[@angelopoulos2023gentle; @ding2023classconditional; @liu2026longtail; @althani2026extreme].

**Supplementary Table S1. Task-equal empirical support and finite-threshold rates. Predicted
finite-threshold rates use the observed target-pool composition retrospectively.**

| Budget | ≥1/class empirical | ≥5/class empirical | Mondrian finite empirical | Mondrian finite predicted |
|---:|---:|---:|---:|---:|
| 25 | 0.9806 | 0.6917 | 0.3306 | 0.3289 |
| 50 | 0.9983 | 0.8939 | 0.7317 | 0.7392 |
| 100 | 1.0000 | 0.9856 | 0.9100 | 0.9015 |
| 250 | 1.0000 | 1.0000 | 0.9993 | 0.9997 |
| 500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

<!-- SID:S-S3-03 -->
Across all H1 strata, predicted and empirical finite-threshold means were 0.779157 and 0.779643,
the mean absolute split–model-stratum rate gap was 0.010973, and the interval-coverage rate was
97.6%. The larger
gap at budget 25 reflects the higher sampling variation of a classwise event. Prospective planning
cannot substitute the true target composition used in this retrospective validation. The frozen
planning analysis instead uses the prespecified prevalence range [0.01, 0.99], which is deliberately
wider than the observed task prevalences and should not be presented as a point prediction for the
six tasks.

## S4. H2 model-stratified effects

<!-- SID:S-S4-01 -->
The main H2 table reports an equal mean of the logistic-regression and CPU-XGBoost task-equal rows.
Supplementary Table S2 shows the endpoint rows separately. Directions agree closely for sigmoid and
isotonic, while the budget-25 intercept means straddle zero. This model separation supports the
aggregate ordering without treating the two models as identical. The complete 40-row
positive-budget table, including every intermediate budget, is supplied as
`source_data_v2/supp_table_s2_h2_model_strata.csv`.

**Supplementary Table S2. Model-stratified primary-set endpoint effects.**

| Model | Method | Budget | Task-equal Δ log loss | Material harm | Estimable |
|---|---|---:|---:|---:|---:|
| Logistic regression | Intercept MLE | 25 | −0.000614 | 44.2% | 97.7% |
| Logistic regression | Intercept MLE | 500 | +0.008617 | 1.3% | 100.0% |
| CPU XGBoost | Intercept MLE | 25 | +0.000409 | 42.0% | 97.7% |
| CPU XGBoost | Intercept MLE | 500 | +0.008989 | 0.8% | 100.0% |
| Logistic regression | Jeffreys intercept | 25 | −0.000139 | 44.9% | 100.0% |
| Logistic regression | Jeffreys intercept | 500 | +0.008497 | 1.3% | 100.0% |
| CPU XGBoost | Jeffreys intercept | 25 | +0.000823 | 42.2% | 100.0% |
| CPU XGBoost | Jeffreys intercept | 500 | +0.008864 | 0.7% | 100.0% |
| Logistic regression | Sigmoid | 25 | −0.026387 | 74.9% | 97.7% |
| Logistic regression | Sigmoid | 500 | +0.007888 | 1.8% | 100.0% |
| CPU XGBoost | Sigmoid | 25 | −0.024515 | 73.0% | 97.7% |
| CPU XGBoost | Sigmoid | 500 | +0.009085 | 1.6% | 100.0% |
| Logistic regression | Isotonic | 25 | −0.439859 | 99.8% | 64.5% |
| Logistic regression | Isotonic | 500 | −0.017208 | 73.3% | 100.0% |
| CPU XGBoost | Isotonic | 25 | −0.403667 | 99.7% | 64.5% |
| CPU XGBoost | Isotonic | 500 | −0.017238 | 73.0% | 100.0% |

<!-- SID:S-S4-02 -->
Harm rates in Table S2 are conditional on successful finite paired effects. Their denominators
therefore differ from availability denominators when class support fails. Isotonic at budget 25 is
the clearest example: approximately 64.5% of attempts were estimable, and almost all successful
pairs were materially harmful. In contrast, Jeffreys smoothing was available in all attempts, but
its endpoint performance stayed close to intercept-only MLE. Neither result supports replacing the
separate availability and conditional-performance columns with a single score.

## S5. H3 budget fractions, deletion checks, and College stress case

<!-- SID:S-S5-01 -->
At absolute budget 100, the frozen H3 fraction is calculated from the actual split-level target
adaptation pool, not by dividing by a conservative registry capacity. The fractions span more than
an order of magnitude among the primary five-task set and approach the entire College pool. These values show
why an absolute count and a relative fraction should both be reported while neither is treated as a
complete task-normalized sample-size measure.

**Supplementary Table S3. Budget 100 as a fraction of the executed target adaptation pool.**

| Task | Analysis scope | Pool fraction | Percent |
|---|---|---:|---:|
| BRFSS Diabetes | primary set | 0.000872 | 0.087% |
| ACS Income | primary set | 0.002958 | 0.296% |
| Diabetes Readmission | primary set | 0.006629 | 0.663% |
| ACS Food Stamps | primary set | 0.011357 | 1.136% |
| NHANES Lead | primary set | 0.019697 | 1.970% |
| College Scorecard | separate stress | 0.961538 | 96.154% |

<!-- SID:S-S5-02 -->
The frozen deletion audit retained 169/170 key LOTO directions and 136/136 LOFO directions. The
only reversal was approximately −0.000028, about 72-fold smaller than the prespecified 0.002
material-benefit threshold. These checks recompute the same frozen paired evidence after removing one task
or family; they do not create independent shifted populations. Complete unrounded tables are
included as `source_data_v2/supp_full_loto.csv` and `source_data_v2/supp_full_lofo.csv`.

**Supplementary Table S4. Direction retention under deletion checks.**

| Check | Retained | Total | Rate |
|---|---:|---:|---:|
| Leave one task out | 169 | 170 | 0.9941 |
| Leave one source family out | 136 | 136 | 1.0000 |

<!-- SID:S-S5-03 -->
College Scorecard supplies a boundary rather than an additional common-task replication. The
executed curve contains only budgets 0, 25, 50, and 100, while 250 and 500 remain unavailable. At
the same method and budget, logistic regression and XGBoost could have opposing effect directions.
Pooling this task into primary-set averages would therefore mix an almost exhaustive small-pool
sample with much smaller sampling fractions and would create incomplete upper-budget curves. The
complete 240-row College derived table is supplied as `source_data_v2/supp_college_stress.csv`.

## S6. H4 strict-threshold diagnostic

<!-- SID:S-S6-01 -->
H4 uses raw target-final minus source-ID differences. A positive Brier or log-loss difference means
deterioration, whereas a positive AUROC difference means improved discrimination. The strict
thresholds exclude equality. All 17 pairs within 0.01 and all 32 pairs within 0.02 worsened on both
proper scores. Other diagnostics were less uniform: at the 0.02 threshold, AUPRC decreased in 3/32,
absolute calibration-intercept error worsened in 28/32, and slope error worsened in 11/32.

**Supplementary Table S5. H4 strict-AUROC-threshold subsets.**

| Strict | Pairs | Brier worse | Log loss worse | Median Δ Brier | Median Δ log loss | AUPRC down | Slope error worse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $\lvert\Delta\mathrm{AUROC}\rvert < 0.01$ | 17 | 17 | 17 | +0.027073 | +0.071711 | 0 | 5 |
| $\lvert\Delta\mathrm{AUROC}\rvert < 0.02$ | 32 | 32 | 32 | +0.018232 | +0.046746 | 3 | 11 |

<!-- SID:S-S6-02 -->
Across all 54 primary pairs, Pearson correlations between AUROC change and Brier/log-loss change
were −0.907897/−0.895421 and Spearman correlations −0.761997/−0.772365. Because multiple design
pairs arose from the same tasks, these are descriptive associations rather than population-level
correlation estimates. The supported diagnostic is narrower: a small AUROC change
did not rule out proper-score deterioration. This separation between ranking and calibration is
already established in shifted clinical and uncertainty-evaluation work [@davis2017aki;
@levy2022autoupdating; @guo2023ehrshift; @ovadia2019trust; @vancalster2019achilles].

## S7. H5 low-budget thresholds and doubletons

<!-- SID:S-S7-01 -->
The primary H5 analysis compares standard and Mondrian conformal prediction on identical task,
split, model, model seed, adaptation seed, budget, score source, evaluation split, and sample hash.
Metric differences are defined only for jointly successful pairs and are summarized by the median
within each task–model–budget cell before equal averaging across cells. At budget 25, the primary
joint-success conditional mean difference in worst-class coverage was +0.1285 and the mean
set-size difference was +0.3280; this summary retains successful Mondrian outputs with infinite
thresholds. Across available attempts, the Mondrian finite-threshold rate was only 30.1%.
Restricting additionally to jointly successful pairs with finite Mondrian thresholds reduced the
mean differences to about +0.0275 for worst-class coverage and +0.0013 for set size. That
sensitivity subset contained four tasks and eight task–model cells; NHANES contributed no finite
row at budget 25. Infinite thresholds and full two-label sets therefore account for much of the
larger primary conditional difference. Non-estimable attempts were not assigned zero metric
differences.

**Supplementary Table S6. Budget-25 H5 sensitivity to finite thresholds.**

| Subset | Δ worst-class coverage | Δ mean set size | Finite-threshold rate in subset |
|---|---:|---:|---:|
| Joint-success pairs (infinite thresholds retained) | +0.1285 | +0.3280 | 0.301 across available attempts |
| Joint-success, finite-Mondrian-threshold pairs | approximately +0.0275 | approximately +0.0013 | 1.000 within this subset |

<!-- SID:S-S7-02 -->
From budget 50 onward, the finite-threshold sensitivity subset still showed a material
coverage–set-size trade-off, and by 250 it approached the primary joint-success conditional result
because finite thresholds were nearly complete. Across all budgets, 50/50 task–model–budget cell
medians had positive worst-class differences, 38/50 were at
least +0.10, and the two model directions agreed in 25/25 task–budget comparisons. Marginal coverage
and absolute marginal-coverage error did not have a common improvement direction. The 50 primary
cell rows are supplied in `source_data_v2/supp_h5_common50_cells.csv`.

## S8. Simulation–natural-shift alignment

<!-- SID:S-S8-01 -->
The probability simulation crossed seven mechanisms, four prevalences, five budgets, and four
calibration methods, with 200 repetitions in each cell. Equal averaging of the frozen isotonic
summary cells gave mean log-loss benefits −0.4102, −0.1721, +0.0583, +0.1973, and +0.2397 at budgets
25, 50, 100, 250, and 500. The sign change contrasts with the primary-set natural-shift aggregate,
which remained negative at 500. The mechanism grid is therefore simpler than the natural shifts in
a scientifically consequential way.

**Supplementary Table S7. Isotonic probability-simulation mean effect.**

| Budget | Mean simulated Δ log loss | Alignment with primary-set real aggregate |
|---:|---:|---|
| 25 | −0.4102 | same negative direction |
| 50 | −0.1721 | same negative direction |
| 100 | +0.0583 | disagrees in sign |
| 250 | +0.1973 | disagrees in sign |
| 500 | +0.2397 | disagrees in sign |

<!-- SID:S-S8-02 -->
The conformal simulations support a class-asymmetry mechanism and the possibility that classwise
coverage requires larger sets. Their stored `finite_threshold_rate` is conditional on estimable
successes; the corresponding all-attempt rates were approximately 0.236, 0.424, 0.516, 0.728, and
0.763 over budgets 25–500. They therefore cannot be directly differenced from the natural-shift
available-attempt H5 rates. Some reference or overconfidence scenarios raised set cost without material
worst-class benefit. Simulation supplies mechanism context only and is not a substitute for an
additional natural task.

**Supplementary Table S8. Public data provenance, version, and redistribution boundary.**

| Task family | Public source/version | Entity and outcome audit key | Access/redistribution boundary |
|---|---|---|---|
| ACS PUMS | 2018 1-year person and housing CSV files | person or `RELP=0` household reference row; `PINCP`/`FS` | public Census microdata; cite source and do not imply Census endorsement |
| BRFSS | 2015, 2017, 2019, 2021 annual files | respondent; valid diabetes and `_PRACE1` codes only | public CDC files; follow CDC citation and use terms |
| NHANES | 1999–2000 through 2017–2018 public-use cycles | cycle-qualified `SEQN`; `LBXBPB` | public-use files under the NCHS data-use agreement [@nchs2024datauseragreement] |
| UCI diabetes | DOI-bound dataset version, accessed for the frozen build | stable-hash patient index encounter; `readmitted != NO` | CC BY 4.0; attribution required [@clore2014diabetes130] |
| College Scorecard | `MERGED2018_19_PP.csv` from the frozen raw-data archive | `UNITID`; `C150_4` | public Department of Education data; release metadata and attribution retained |

## S9. Reproducibility and source-data map

<!-- SID:S-S9-01 -->
Formal computation used Python 3.11.15, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1,
scikit-learn 1.9.0, XGBoost 3.2.0, matplotlib 3.11.0, seaborn 0.13.2, PyArrow 24.0.0,
PyYAML 6.0.3, psutil 7.2.2, and Folktables 0.0.12. The server ran CentOS Linux 8 with kernel
4.18.0-348.el8.x86_64, two Intel Xeon Silver 4210R processors (20 physical and 40 logical cores)
and approximately 125 GiB RAM. Formal XGBoost execution was CPU-only with `n_jobs=4`. The exact
package lock is `environment.lock.yml`; a user-space reconstruction command is
`micromamba create --prefix ~/.local/envs/reliable-shift-stage1 --file environment.lock.yml`.
The supplied exact-build lock is for `linux-64`. Document conversion used Pandoc 3.10 and Tectonic
0.16.9; PDF inspection used Poppler 26.05.0. These presentation-only tools are recorded separately
in `DOCUMENT_TOOLCHAIN.lock.md` and did not fit models or alter experiment outputs.

<!-- SID:S-S9-02 -->
The base-model and update details needed to re-execute the prespecified model configurations from
prepared inputs are fully specified in
the main Methods. In particular, logistic-regression candidates used $C=0.1$ and 1.0; XGBoost
candidates used `(min_child_weight, reg_lambda)` equal to `(1,1.0)` and `(5,2.0)`; probabilities
were clipped at $10^{-6}$ only for numerical stability; intercept offsets used Brent roots on
`[-40,40]`; and sigmoid/isotonic solver settings are reported verbatim. Split seeds were 0–2,
logistic model seed was 0, XGBoost model seeds were 0–1, and adaptation seeds were 0–99.

<!-- SID:S-S9-03 -->
Each main figure has both PDF and 400-dpi PNG output. Every figure and main table has a CSV source
file, and `source_data_v2/SOURCE_DATA_INDEX.csv` records its SHA-256 and transformation. Figure 1 is a
protocol diagram; Figure 2 uses the 20 equal-model H2 method–budget summaries; Figure 3 contains the
200 unpooled primary-set task–model–method–budget H3 cells; Figure 4 uses the five archived H5 budget
records; and Figure 5 contains the 54 primary H4 pairs. Main Tables 1–4 have independent CSV,
Markdown, and LaTeX forms.

<!-- SID:S-S9-04 -->
The public release contains manuscript files, figures, tables, figure- and table-level source data,
analysis and artifact-generation code, environment specifications, checksums, and a documented
workflow for one representative end-to-end ACS model shard. An optional workflow re-executes the
54 prespecified model configurations from prepared, audited Parquet inputs. The package does not
redistribute the original third-party datasets, reconstruct five of the six task families from raw
official downloads, or contain the complete intermediate results tree and downstream analysis
pipeline needed to regenerate all article source-data files from that re-execution alone.
Data-acquisition instructions identify the official sources and applicable provider terms.

## Supplementary References

::: {#refs}
:::
