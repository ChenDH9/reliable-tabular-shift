---
author:
- Dinghao Chen
bibliography: ../references/references.bib
lang: en-US
link-citations: true
title: Complexity and Label-Budget Trade-offs in Probability Updating
  and Class-Conditional Conformal Prediction under Natural Tabular
  Shifts
---

# Complexity and Label-Budget Trade-offs in Probability Updating and Class-Conditional Conformal Prediction under Natural Tabular Shifts

**Author:** Dinghao Chen

**Affiliation:** College of Big Data, Baoshan University, Baoshan,
Yunnan 678000, China

**Corresponding author:** Dinghao Chen

**Submission administrator:** Dinghao Chen

**ORCID:** None

*Author Cover Page — the entries above must exactly match the PeerJ
online submission record.*

## Abstract

<!-- SID:ABS01 -->

After transfer across populations, regions, or time, a predictive model
can retain discrimination while its probability estimates deteriorate—a
failure mode that discrimination metrics alone cannot reveal. We asked
whether limited target labels acquired by label-blind random sampling
could reliably support probability updating and class-conditional
conformal prediction under natural tabular shifts. Across five
natural-shift tasks, two base models, four probability-updating methods,
and absolute budgets of 25–500 labels, each target-fitted update was
compared with its same-method source-fitted baseline; non-estimable
attempts were retained in feasibility accounting. Increasing the budget
improved feasibility where it was initially limited and reduced
observed harm, but label count alone did not ensure benefit across
tasks, models, and calibrators. The task- and model-equal mean log-loss
benefit for intercept-only updating increased from −0.0001 at 25 labels
to +0.0088 at 500, while its material-harm rate among successful finite
pairs fell from 43.1% to 1.0%. Sigmoid's mean benefit changed from
−0.0255 to +0.0085, and its corresponding harm rate fell from 74.0% to
1.7%. Isotonic remained harmful on average throughout the tested grid:
mean benefit changed from −0.4218 to −0.0172, while its corresponding
harm rate fell from 99.7% to 73.2%. No method at any tested absolute
budget met the prespecified +0.002 log-loss improvement criterion in all
10 task–model cells. Among jointly successful Standard–Mondrian pairs,
the within-cell median Mondrian-minus-Standard difference in observed
worst-class coverage was positive in all 50 task–model–budget cells, but
mean prediction-set size also increased, and only 30.1% of available
Mondrian attempts yielded finite thresholds at 25 labels. In the
finite-threshold sensitivity analysis at 25 labels (8/10 task–model
cells), the equal-cell mean worst-class-coverage difference was +0.0275
rather than +0.1285, and the mean set-size difference was +0.0013 rather
than +0.3280. These findings argue against a universal label-count rule:
within the evaluated grid, reliability was specific to task, model,
method, class support, and budget.

**Keywords:** probability calibration; model updating; distribution
shift; tabular data; conformal prediction; class-conditional coverage;
label budget; reliability

## 1. Introduction

<!-- SID:I01 -->

Predictive models deployed across populations, regions, or time periods
may retain useful ranking performance while their probability estimates
cease to reflect target-domain outcomes. Reliable probabilities matter
whenever predictions inform thresholds, resource allocation, or risk
communication, so a modest set of newly observed target labels is often
used to update a model without rebuilding it. Clinical and broader shift
studies have documented calibration drift, heterogeneous changes across
models, and the need to assess probability quality separately from
discrimination ([Davis et al. 2017](#ref-davis2017aki); [Levy et al.
2022](#ref-levy2022autoupdating); [Ovadia et al.
2019](#ref-ovadia2019trust); [Van Calster et al.
2019](#ref-vancalster2019achilles)). The practical question is therefore
not only whether a source model transfers, but whether limited labeled
target data can improve its reliability.

<!-- SID:I02 -->

Such updating is not automatically beneficial. A method may be
unavailable when a blind sample contains too few observations from one
class, and an estimable update may still increase log loss or Brier
score through variance or overfitting. Earlier work already shows that
recalibration behavior depends on sample size and method complexity:
parsimonious prediction-model updates can be preferable with scarce
data, while flexible procedures need more support and can harm an
already well-calibrated predictor ([Steyerberg et al.
2004](#ref-steyerberg2004updating); [Niculescu-Mizil and Caruana
2005](#ref-niculescu2005goodprob); [Ojeda et al.
2023](#ref-ojeda2023comparison)). Class-conditional conformal procedures
face a related feasibility constraint because finite classwise
thresholds require sufficient calibration support in each class ([Ding
et al. 2023](#ref-ding2023classconditional); [Liu et al.
2026](#ref-liu2026longtail)).

<!-- SID:I03 -->

Label budget, target prevalence, base model, and update complexity
consequently need to be considered together. The same total budget can
contain very different numbers of minority-class labels across targets,
and the same number of labels can represent very different fractions of
the available target population. Moreover, the data needed by an
intercept correction differ from those needed by sigmoid or isotonic
calibration, just as the data needed for a marginal conformal threshold
differ from those needed for separate classwise thresholds.
Finite-sample recalibration and model-updating studies have established
these dependencies in synthetic, clinical, or independently calibrated
settings ([Sun et al. 2023](#ref-sun2023minimumrisk); [Vergouwe et al.
2017](#ref-vergouwe2017closed); [Davis et al.
2020](#ref-davis2020protocols)). What remains operationally important is
their joint behavior under a fixed natural-shift protocol.

<!-- SID:I04 -->

Existing research addresses the constituent questions from several
directions. Probability-calibration studies compare parametric and
nonparametric updates; model-maintenance studies examine recalibration
under temporal or external validation; tabular-shift benchmarks
characterize heterogeneous geographic, temporal, and population shifts;
and conformal work studies marginal, classwise, and sparse-class
coverage ([Ojeda et al. 2023](#ref-ojeda2023comparison); [Gardner et al.
2023](#ref-gardner2023tableshift); [Liu et al.
2023](#ref-liu2023whyshift); [Ding et al.
2023](#ref-ding2023classconditional)). Other work adapts calibration or
uncertainty under explicit covariate- or label-shift assumptions,
sometimes without labeled target outcomes ([Park et al.
2020](#ref-park2020calibrated); [Popordanoska et al.
2024](#ref-popordanoska2024lascal); [Podkopaev and Ramdas
2021](#ref-podkopaev2021labelshift)).

<!-- SID:I05 -->

This study addresses a joint empirical question: how benefit, harm,
estimability, budget heterogeneity, and prediction-set cost interact
under a common label-blind protocol for natural tabular domain shifts.
We evaluate public binary tabular tasks with one record per entity,
acquire target labels by label-blind random sampling, and compare
updates against same-method budget-zero source baselines. This design
keeps failed or unavailable cells visible and connects probability
adaptation with conformal feasibility and efficiency. The intended
contribution is empirical and protocol-based rather than methodological:
a bounded quantitative map for the evaluated setting, not a general
ranking of calibrators or a deployment rule.

<!-- SID:I06 -->

One primary contribution concerns complexity, budget, benefit, and harm.
Across the primary five-task set, logistic regression and XGBoost base
models, and label-blind random target-label budgets from 25 to 500, the
mean benefit of intercept updating became positive at a lower tested
budget than sigmoid recalibration. Isotonic calibration retained high
observed material-harm rates among successful finite pairs throughout
the prespecified test grid. This result describes the observed ordering
and harm frequencies within these tasks, models, methods, and budgets;
it does not establish a general calibrator ranking.

<!-- SID:I07 -->

The second primary contribution concerns budget heterogeneity. Within
the same five-task, two-model, four-calibrator grid, no tested absolute
target-label budget produced a median log-loss benefit of at least 0.002
in every task–model–calibrator cell. The fractions of target adaptation
pools represented by a given absolute budget varied substantially; the
design did not test a common relative-budget grid and therefore does not
identify a universal fraction threshold. Leave-one-task-out and
leave-one-source-family-out analyses assess how strongly this conclusion
depends on individual tasks or data families, while a separate
small-domain stress case illustrates why nominal label counts cannot be
interpreted without the size and class composition of the target pool.

<!-- SID:I08 -->

The secondary contribution concerns class-conditional reliability. Among
jointly successful paired comparisons over the primary five-task set,
two models, and label-blind random budgets from 25 to 500, the median
Mondrian-minus-standard difference in worst-class coverage was positive
in every evaluated task–model–budget cell. The increase was accompanied
by budget-dependent finite-threshold availability and larger prediction
sets, including more binary doubletons at smaller budgets. We therefore
report conditional metric differences, estimability, finite-threshold
rates, and set cost together. Auxiliary analyses document eligibility
boundaries and show that small AUROC changes did not rule out worse
proper scores, but neither auxiliary result is treated as a central
contribution.

## 2. Related Work

### 2.1 Probability calibration and model updating

<!-- SID:RW01 -->

Post-hoc probability calibration includes parametric corrections such as
logistic or sigmoid scaling and more flexible procedures such as
isotonic regression. Comparative work has long shown that performance
depends on both the underlying learner and the amount of calibration
data. Niculescu-Mizil and Caruana found sample-dependent ordering
between Platt scaling and isotonic regression and observed that
recalibration can harm models that are already well calibrated
([Niculescu-Mizil and Caruana 2005](#ref-niculescu2005goodprob)).
Broader comparisons likewise show heterogeneity across probability
models, structural differences, prevalence, and calibration sample size;
regression-based approaches often behave more stably when data are
limited, whereas isotonic behavior is especially sample-sensitive
([Ojeda et al. 2023](#ref-ojeda2023comparison)). Finite-sample analyses
further connect calibrator complexity with risk and show how the useful
resolution of a recalibration method should change with available data
([Sun et al. 2023](#ref-sun2023minimumrisk)). These studies directly
motivate, and constrain the interpretation of, our complexity–budget
analysis.

<!-- SID:RW02 -->

Prediction-model updating research provides an equally direct precedent.
Clinical external-validation studies compare retaining a model with
adjusting its intercept, revising its calibration slope, or changing
more coefficients. Their results favor matching update complexity to
target sample size and using shrinkage or staged selection to control
overfitting ([Steyerberg et al. 2004](#ref-steyerberg2004updating);
[Vergouwe et al. 2017](#ref-vergouwe2017closed)). Longitudinal
maintenance studies also show that the timing and extent of updating can
differ by algorithm, so a single updating policy need not serve every
model ([Davis et al. 2020](#ref-davis2020protocols); [Levy et al.
2022](#ref-levy2022autoupdating)). Recent work under natural image
shifts similarly reports heterogeneous trade-offs between simple
post-hoc and shift-specific calibration procedures ([Roschewitz et al.
2025](#ref-roschewitz2025posthoc)). Our study does not propose a
calibration operator. Its narrower increment is a paired
benefit-and-harm audit of prespecified calibrators under blind
target-label budgets, with unavailable fits retained rather than
silently removed.

### 2.2 Natural tabular distribution shift

<!-- SID:RW03 -->

Natural tabular shifts differ from random train–test splits because
domain membership can represent geography, calendar time, institution,
or population group. TableShift organizes public binary tabular tasks
around such naturally occurring domains and documents substantial
heterogeneity in out-of-domain performance ([Gardner et al.
2023](#ref-gardner2023tableshift)). WhyShift complements benchmark-style
evaluation with a language for describing shift structure and shows that
interventions can behave differently across datasets and shift types
([Liu et al. 2023](#ref-liu2023whyshift)). These works establish both
the natural tabular setting and the importance of dataset-specific
interpretation. The present study uses that setting for a different
endpoint: recovery of probability and prediction-set reliability with a
limited, label-blind target sample.

<!-- SID:RW04 -->

Reliability under shift has also been studied under more specific
assumptions or in other modalities. Covariate-shift methods use
importance weighting or domain adaptation to transfer calibration
without labeled target outcomes ([Park et al.
2020](#ref-park2020calibrated)), while label-shift methods estimate or
correct target calibration under assumptions about changing class
proportions ([Popordanoska et al. 2024](#ref-popordanoska2024lascal);
[Podkopaev and Ramdas 2021](#ref-podkopaev2021labelshift)). Medical
studies show that natural temporal or population shifts can affect
discrimination and calibration differently and that reliability updates
may persist across clinically relevant subpopulations ([Davis et al.
2017](#ref-davis2017aki); [Guo et al. 2023](#ref-guo2023ehrshift);
[Kapash et al. 2025](#ref-kapash2025multiaccuracy)). These approaches
answer important but distinct questions. Our protocol does not assume
that the observed natural shifts are pure covariate or label shift; it
instead measures what happens when a prespecified number of target
outcomes is sampled without observing labels during acquisition.

### 2.3 Conformal prediction and class-conditional coverage

<!-- SID:RW05 -->

Conformal prediction converts model scores and a labeled calibration
sample into prediction sets with finite-sample coverage properties under
its stated sampling conditions ([Angelopoulos and Bates
2023](#ref-angelopoulos2023gentle)). Adaptive scores can improve set
efficiency ([Romano et al. 2020](#ref-romano2020aps)), and weighting or
sequential adaptation can address particular forms of covariate shift or
nonexchangeability ([Tibshirani et al.
2019](#ref-tibshirani2019weighted); [Foygel Barber et al.
2023](#ref-barber2023beyond); [Gibbs and Candès
2024](#ref-gibbs2024online)). Few-shot conformal work also shows that
very small calibration samples can yield uninformative sets and studies
how auxiliary tasks may improve efficiency ([Fisch et al.
2021](#ref-fisch2021fewshot)). Our setting instead uses no
auxiliary-task transfer and treats the finite or infinite status of each
threshold as an observed protocol outcome.

<!-- SID:RW06 -->

Marginal coverage can conceal weak performance for infrequent classes,
motivating class-conditional or Mondrian procedures. Prior work
establishes that rare classes may lack enough calibration observations
for finite classwise quantiles and that better classwise coverage can
require larger prediction sets ([Ding et al.
2023](#ref-ding2023classconditional); [Liu et al.
2026](#ref-liu2026longtail)). Recent methods seek different points on
this coverage–efficiency trade-off through augmented ranks, prevalence
adjustment, interpolation, or hierarchical fallback ([Shi et al.
2024](#ref-shi2024classwise); [Ding et al.
2026](#ref-ding2026longtailed); [Papaioannou et al.
2026](#ref-papaioannou2026robust)). Binary anomaly-detection evidence
likewise shows that class-conditional coverage improvements can be
accompanied by overcoverage or efficiency costs under extreme imbalance
([Althani 2026](#ref-althani2026extreme)). Accordingly, our
Standard–Mondrian comparison is a budget-response audit of an
established trade-off, not an overall method ranking.

### 2.4 Positioning of the present study

<!-- SID:RW07 -->

Prior work has examined complexity-sensitive recalibration,
algorithm-dependent model updating, natural tabular shifts, calibration
under explicit shift assumptions, sparse-class conformal thresholds,
coverage–set-size trade-offs, and the distinction between discrimination
and probability reliability. This study integrates these components in
one prespecified empirical protocol that jointly reports benefit, harm,
estimability, target-pool fraction, and prediction-set cost across the
primary five-task set, two base models, target-label budgets of 25–500,
and four probability updates. The contribution is empirical and
protocol-based rather than methodological or theoretical.

<!-- SID:RW08 -->

No complete duplicate of that full protocol was identified in the
bounded literature audit, but this observation is limited to the
searched and verified record and is not evidence of global absence. The
closest studies overlap substantially with individual questions while
differing in data modality, shift assumptions, sample regime, adaptation
information, or reported outcomes. We therefore organize the manuscript
around scoped quantitative boundaries: when evaluated calibrators became
beneficial or harmful, whether one budget satisfied the prespecified
criteria across the tested grid, and how class-conditional coverage
changed with threshold feasibility and set size. All conclusions remain
conditional on the selected tasks, models, methods, acquisition rule,
and budgets.

## 3. Study Design and Scope

<!-- SID:SD01 -->

We studied a constrained target-adaptation setting in which a
probabilistic binary classifier had already been trained in a source
domain and a limited number of labels could subsequently be acquired
from a shifted target domain. The predictor itself remained frozen. The
decision under study was therefore not whether to retrain the predictive
model, but whether a particular probability or conformal update was
usable and beneficial at a given target-label budget. Label acquisition
was uniform and label-blind: target outcomes were revealed only after
the sampled entities had been selected. This design separates adaptation
from active learning, oracle stratification, and outcome-aware
monitoring.

<!-- SID:SD02 -->

The empirical scope comprised six public binary tabular tasks from five
source families. Five tasks formed the common analysis set because every
prespecified budget from 25 to 500 labels was available. College
Scorecard was retained as a separate small-domain stress case because
its target adaptation capacity was 107 entities; budgets 250 and 500
were recorded as unavailable rather than imputed, refilled, or
extrapolated. The primary evidence concerned probability-update
complexity and budget (H2) and the absence of a common tested absolute
budget across tasks and methods (H3). The Standard–Mondrian
class-conditional coverage trade-off (H5) was secondary. Class-support
feasibility (H1), source-to-target metric changes (H4), and simulations
were auxiliary. H6 was designated `not_in_final_scope`; no
sampling-policy comparison was conducted.

<!-- SID:SD03 -->

Repetition represented three distinct sources of variation. Entity
partitions were generated for split seeds 0, 1, and 2. Logistic
regression used model seed 0, whereas CPU XGBoost used model seeds 0
and 1. Within each resulting task–split–model shard, positive-budget
acquisition used 100 adaptation seeds. They were numbered 0–99, derived
from the configured count. The budget-zero source reference used
adaptation seed −1. This produced 54 prespecified model configurations:
six tasks, three split seeds, and three model/seed combinations per
split. Split, model, and adaptation seeds were stored separately and
were not treated as interchangeable replicates.

<!-- SID:SD04 -->

The study design, archived experiment results, independent scientific
analysis, and allowed claims were fixed before manuscript preparation;
their version identifiers and hashes are reported in the supplement and
repository audit. Manuscript figures and tables use only these archived
results. Subsequent reproduction checks were executed separately and did
not replace or modify the outputs used for the reported analyses.

<!-- SID:SD05 -->

The unit of a probability decision was a
task–base-model–calibrator–budget combination supported by paired runs,
rather than an isolated run or a pooled collection of rows. The unit of
the conformal decision was the same task and base model at a given
budget, with Standard and Mondrian evaluated on an identical target
sample and score source. This distinction matters because a large task
or a model with two random seeds would otherwise contribute more rows
and dominate a pooled mean. The design instead asks whether the
direction and risk profile remain visible after collapsing repeated
acquisitions within tasks and then weighting scientific units equally.
It also treats usability as part of the decision: a candidate update is
described by availability, eligibility, conditional performance, and,
for Mondrian, finite-threshold status. These are complementary
coordinates rather than a sequence in which missing cells are discarded
before accuracy is summarized.

![Study design and the seven mutually exclusive data roles. Target
adaptation and final evaluation remain entity-disjoint; all methods
within a run share a label-blind random nested target
sample.](../figures/figure1_study_design.png){#fig:study-design width=100%}

## 4. Data and Natural Domain Shifts

<!-- SID:DATA01 -->

The primary five-task set contained two 2018 American Community Survey
(ACS) 1-year Public Use Microdata Sample tasks and one task each from
the Behavioral Risk Factor Surveillance System (BRFSS), the National
Health and Nutrition Examination Survey (NHANES), and the UCI
diabetes-readmission data ([U.S. Census Bureau
2018](#ref-uscensus2018acspums), [2019](#ref-uscensus2018pumsdict);
[Clore et al. 2014](#ref-clore2014diabetes130)). ACS Income used the
Folktables adult filter (age above 16 years, personal income above
\$100, positive usual weekly hours, and person weight at least one),
labeled personal income `PINCP` at least \$56,000 as positive, and
contrasted New England (`DIVISION = 1`, target) with other census
divisions (source). This threshold is a project adaptation of the
benchmark rather than an exact reproduction of the strict
`PINCP > 56,000` TableShift task ([Ding et al.
2021](#ref-ding2021folktables); [Gardner et al.
2023](#ref-gardner2023tableshift)). ACS Food Stamps selected the
household reference-person row (`RELP = 0`), ages 18–61, personal—not
household—income `PINCP` at most \$30,000, and `HUPAC` codes 1–4, then
predicted household food-stamp receipt (`FS = 1`). Those `HUPAC` codes
include its no-children category; the cohort must not be described as
restricted to households with children. East South Central
(`DIVISION = 6`) was the target and all other divisions were the source.

<!-- SID:DATA02 -->

BRFSS combined the 2015, 2017, 2019, and 2021 public-use files ([Centers
for Disease Control and Prevention 2015](#ref-cdcbrfss2015),
[2017](#ref-cdcbrfss2017), [2019](#ref-cdcbrfss2019),
[2021](#ref-cdcbrfss2021)). `DIABETE3` (2015/2017) or `DIABETE4`
(2019/2021) code 1 was positive; codes 2 (pregnancy only), 3 (no), and 4
(prediabetes/borderline) were negative, and refused, unknown, and
missing responses were excluded. `_PRACE1 = 1` (White race) defined the
source; codes 2–6 (Black or African American, American Indian or Alaska
Native, Asian, Native Hawaiian or other Pacific Islander, or other race)
defined the target. This is a race-code partition, not a
White-non-Hispanic contrast, and `_PRACE1` and the diabetes item were
excluded from predictors. NHANES merged cycle-qualified respondent
identifiers across cycles from 1999–2000 through 2017–2018 and retained
complete public-use `LBXBPB`, `INDFMPIR`, and `RIDAGEYR` records at age
18 or younger ([National Center for Health Statistics
2020a](#ref-nchsnhanes1999_2018), [2020b](#ref-nchsnhanes2017demo),
[2020c](#ref-nchsnhanes2017lead)). The target had family poverty-income
ratio at most 1.3, the source had a ratio above 1.3, and the positive
label was blood lead at least 3.5 μg/dL. The 3.5 μg/dL value was used as
a benchmark cutpoint informed by the CDC blood lead reference value, not
as a toxicity or clinical-diagnosis threshold; the age range also
extends beyond the 1–5-year population used to derive that reference
value ([Ruckart et al. 2021](#ref-ruckart2021blrv)). Public-use lead
measurement availability was not uniform over cycles, and no survey
weighting was used.

<!-- SID:DATA03 -->

In the UCI task, `admission_source_id = 7` (emergency room) defined the
target and all other retained admission sources defined the source. The
positive label was `readmitted != NO`, so both `<30` and `>30`
categories were positive; it is not the original study's narrower 30-day
endpoint ([Clore et al. 2014](#ref-clore2014diabetes130); [Strack et al.
2014](#ref-strack2014hba1c)). A deterministic stable hash selected one
patient index encounter, which is not a chronological first-encounter
rule, and the prediction time was hospital discharge. College Scorecard
used the fixed `MERGED2018_19_PP.csv` release, rather than a generic
calendar-2018 snapshot ([U.S. Department of Education
2026b](#ref-usedcollegescorecard2026),
[2026a](#ref-usedcollegescorecarddict2026),
[2025](#ref-usedcollegescorecarddocs2025)). Public and private nonprofit
institutions (`CONTROL = 1` or 2) formed the source, private for-profit
institutions (`CONTROL = 3`) formed the target, and `C150_4 ≤ 0.50`
denoted completion at or below 50% among the applicable IPEDS
first-time, full-time cohort within 150% of normal program time. It is
not a pooled all-student completion measure. College contributed only
the lower-budget stress analysis. Across all tasks, domain indicators,
direct identifiers, label variables, and audit keys were excluded from
predictors. These observational partitions do not identify causal
effects of geography, race, poverty, admission source, or institutional
control.

Every task obeyed a one-entity–one-row rule. The entities were an ACS
person, an ACS household represented by its reference-person row, a
BRFSS respondent, an NHANES respondent identified within survey cycle, a
UCI patient represented by the stable-hash index encounter, and a
College Scorecard institution. Audits found no duplicate entity in the
final task tables. Entity identity, rather than a potentially repeated
record, was the leakage-control unit.

<!-- SID:DATA04 -->

For each split seed, entities were assigned once to seven mutually
exclusive roles: `source_train`, `source_tune`,
`source_probability_calibration`, `source_conformal_calibration`,
`source_id_test`, `target_adaptation_pool`, and `target_final_test`. A
seed-specific hash allocated source entities approximately
60%/10%/10%/10%/10% to the five source roles and target entities 40%/60%
to adaptation and final testing, after which prespecified label-blind
caps were applied. Maximum source-role sizes were 100,000 for training,
20,000 each for tuning and the two calibration roles, and 30,000 for
source in-domain testing; target final testing was capped at 50,000,
while the target adaptation pool was uncapped. Smaller roles retained
all available entities. Model selection used only source training and
tuning. The two source calibration roles were disjoint, and target-final
labels were used only for evaluation.

<!-- SID:DATA05 -->

Target acquisition used `blind_random` only. For each dataset, split
seed, and adaptation seed, entities in the target adaptation pool were
ordered by a deterministic hash that did not accept labels as input.
Budgets 25, 50, 100, 250, and 500 were nested prefixes of this ordering,
so all methods evaluated at a given run shared the same selected
entities and `sample_hash`. A budget exceeding the pool was marked
`not_available`; there was no resampling after outcomes were observed.
Results were reported on both the absolute budget axis and the fraction
of the available target adaptation pool because identical label counts
represented different sampling intensities across tasks.

<!-- SID:DATA06 -->

All task registries were finalized before the writing stage and linked
to processed-data checksums. Table 1 reports positive-outcome rates in
the audited full source and target domains, whereas the adaptation
experiments used seed-specific entity-disjoint pools derived from those
domains. The two quantities should not be conflated: the full-domain
rate describes the observed shift, while the pool composition determines
the realized support probabilities for a particular split. Public
availability also does not make the sources interchangeable. ACS and
BRFSS are large surveys, NHANES combines survey and laboratory
measurements, the UCI table originates from hospital encounters after
patient-level indexing, and College Scorecard is an institution-level
snapshot. Keeping source family and entity unit explicit prevents row
counts from being interpreted as a single homogeneous sampling frame.

**Table 1. Task definitions, natural domains, entity units, audited
counts, and analysis roles.**

| Task | Source domain | Target domain | Outcome / inclusion | Entity | Period / version | Source n (positive %) | Target n (positive %) | Analysis role |
|----|----|----|----|----|----|----|----|----|
| ACS Income | DIVISION != 1 (outside New England) | DIVISION = 1 (New England) | PINCP ≥ \$56,000 after Folktables adult filter | person | 2018 ACS 1-year PUMS | 1,575,270 (32.9%) | 84,346 (40.3%) | common five |
| ACS Food Stamps | DIVISION != 6 (outside East South Central) | DIVISION = 6 (East South Central) | FS = 1; reference-person row aged 18–61, PINCP ≤ \$30,000, HUPAC 1–4 | household (RELP = 0) | 2018 ACS 1-year PUMS | 276,050 (23.0%) | 21,932 (27.1%) | common five |
| BRFSS Diabetes | \_PRACE1 = 1 (White) | \_PRACE1 = 2–6 (Black/AIAN/Asian/NHPI/Other) | DIABETE3/4 = 1; codes 2–4 negative | respondent | BRFSS 2015, 2017, 2019, 2021 | 1,409,220 (12.6%) | 287,074 (17.0%) | common five |
| NHANES Lead | INDFMPIR \> 1.3 | INDFMPIR ≤ 1.3 | LBXBPB ≥ 3.5 µg/dL among age ≤18 | SEQN within cycle | NHANES 1999–2000 to 2017–2018 | 14,759 (2.6%) | 12,740 (8.0%) | common five; rare-positive |
| Diabetes Readmission | admission_source_id != 7 | admission_source_id = 7 (emergency room) | readmitted != NO (both \<30 and \>30 positive) | patient index encounter | UCI encounters, 1999–2008 | 31,833 (30.7%) | 37,835 (35.7%) | common five |
| College Scorecard | CONTROL = 1 or 2 (public/nonprofit) | CONTROL = 3 (private for-profit) | C150_4 ≤ 0.50 | institution (UNITID) | MERGED2018_19_PP.csv | 1,986 (46.7%) | 268 (66.4%) | separate small-domain stress |

## 5. Methods

<!-- SID:M01 -->

We evaluated logistic regression and CPU XGBoost as two contrasting base
predictors. Numeric features were median-imputed and standardized.
Categorical features were one-hot encoded with unknown-category handling
and a minimum category frequency of five. For each split and model seed,
two prespecified hyperparameter candidates were fitted on
`source_train`; the candidate with higher AUROC on `source_tune` was
retained. Logistic regression used the liblinear solver, no class
weighting, a maximum of 1,000 iterations, and $`C\in\{0.1,1.0\}`$.
XGBoost used 120 depth-four trees, learning rate 0.05, row and column
subsampling 0.9, histogram tree construction on CPU, four CPU threads,
and log loss as the evaluation metric. Its two candidates paired minimum
child weight 1 with $`L_2`$ regularization 1.0 and minimum child weight
5 with $`L_2`$ regularization 2.0. Once selected, the entire
preprocessing–prediction pipeline remained fixed.

<!-- SID:M02 -->

Probability updates acted on the base-model probability $`p`$. The
uncalibrated method was the identity mapping and was stored at budget
zero only. The intercept-only maximum-likelihood update added a fitted
scalar offset to $`\operatorname{logit}(p)`$, retaining unit slope. The
Jeffreys-smoothed intercept update used the same unit-slope form but
matched the mean fitted probability to $`(n_1+0.5)/(n+1)`$, permitting a
finite fit even when the sampled target labels belonged to one class.
Sigmoid calibration fitted an unpenalized logistic regression with free
intercept and slope to the base-model log odds. Isotonic calibration
fitted a bounded monotone mapping from the base probability to the
binary outcome. Base probabilities were clipped to
$`[10^{-6},1-10^{-6}]`$ only where logit or log-loss numerical stability
required it; ranking metrics retained the unclipped ordering. Sigmoid
calibration used scikit-learn logistic regression with $`C=\infty`$, the
`lbfgs` solver, and at most 1,000 iterations. Both intercept solvers
used SciPy's Brent root finder on $`[-40,40]`$. Isotonic regression used
`out_of_bounds="clip"`, `y_min=0`, and `y_max=1`.

<!-- SID:M03 -->

Method eligibility was explicit. Intercept-only maximum likelihood and
sigmoid required at least one observed label from each class. Isotonic
required at least 25 labels in total, at least five labels per class,
and at least 10 unique input scores. Jeffreys smoothing required a
nonempty sample but did not require both classes. At budget zero, each
fitted method was trained on `source_probability_calibration`. At a
positive budget, it was instead fitted to the label-blind random target
prefix and evaluated on `target_final_test`. Every adapted probability
method was compared with its own source-fitted budget-zero version; a
shared uncalibrated comparator was not substituted for these same-method
baselines.

<!-- SID:M04 -->

We evaluated Standard split conformal and outcome-class-conditional
Mondrian conformal prediction at nominal coverage 0.90. For candidate
class $`y`$, nonconformity was $`1-p_y`$. Standard conformal used one
threshold from all calibration scores, whereas Mondrian used separate
thresholds for classes 0 and 1 and applied the threshold corresponding
to each candidate label. The finite-sample order was
$`\lceil(n+1)(1-\alpha)\rceil`$. Here $`\alpha=0.10`$; a requested order
above the available calibration size produced an infinite threshold
rather than a silently altered quantile. Mondrian required at least five
calibration labels per class, although estimability did not itself
guarantee that both classwise thresholds were finite. Two fixed score
sources were evaluated: base uncalibrated probabilities and
probabilities transformed by a sigmoid fit on the disjoint source
probability-calibration role.

<!-- SID:M05 -->

Budget-zero conformal thresholds were fitted on
`source_conformal_calibration`. Positive-budget thresholds were
refreshed from the same label-blind random target prefix used by the
probability methods, preserving exact sample pairing. A prediction set
included every class whose candidate-label nonconformity did not exceed
its threshold. For binary outcomes, set sizes therefore ranged from zero
to two; a doubleton was the full two-label set. Standard and Mondrian
results were always paired on dataset, split, model, model seed,
adaptation seed, budget, score source, evaluation split, and sample
hash.

<!-- SID:M06 -->

Every attempted cell retained its computational state. `success` denoted
a completed fit and evaluation. `not_available` denoted a prespecified
budget exceeding the target pool. `not_estimable` denoted failure of a
declared support or method-eligibility rule. `failed` was reserved for a
technical exception. Separate indicators recorded whether a method was
estimable and whether its conformal threshold was finite. These
categories were never collapsed into a single missing value, and
unavailable or non-estimable attempts were not counted as neutral
performance outcomes.

<!-- SID:M07 -->

The methods form a complexity comparison, not a claim that their
parameter counts alone determine risk. Intercept-only MLE estimates one
target offset; the smoothed version modifies its small-sample target;
sigmoid estimates an intercept and slope; and isotonic can adapt a
monotone curve to the observed scores. Their behavior can also depend on
source score geometry, target prevalence, and the base predictor. For
this reason, each adapted row retained the observed positive and
negative counts, number of unique input scores, target-pool fraction,
and exact sample hash. Conformal score sources were likewise fixed
rather than selected after target performance was observed. These
records allow complexity to be discussed alongside its actual support
and sampling context without reducing the experiment to a single nominal
degrees-of-freedom ordering.

**Table 2. Evaluated methods, support rules, and reported outcomes.**

| Method | Family | Frozen support rule | Comparison role | Reported outputs |
|----|----|----|----|----|
| Uncalibrated | Probability reference | budget 0 only | no target fit | AUROC, AUPRC, Brier, log loss, calibration intercept/slope |
| Intercept-only MLE | Probability update | ≥1 label per class | same-method source baseline | Brier, log loss, calibration intercept/slope |
| Jeffreys intercept | Probability update | nonempty sample; smoothed | low-budget availability comparator | Brier, log loss, calibration intercept/slope |
| Sigmoid | Probability update | ≥1 label per class | same-method source baseline | Brier, log loss, calibration intercept/slope |
| Isotonic | Probability update | n≥25; ≥5/class; ≥10 unique scores | same-method source baseline | Brier, log loss, calibration intercept/slope |
| Standard | Conformal | total-size quantile | nominal 0.90 marginal sets | marginal/worst-class coverage, mean set size, doubletons |
| Mondrian | Conformal | ≥5/class to estimate; exact finite-quantile check | nominal 0.90 class-conditional sets | marginal/worst-class coverage, mean set size, doubletons, finite threshold |

### Ethics Statement

This study involved only secondary analysis of publicly available,
de-identified datasets. The author did not recruit or interact with
participants, collect new human-participant data, conduct interventions,
or access direct personal identifiers. On this basis, separate
institutional ethics review was not required, and no approval or
determination number was issued.

## 6. Evaluation Protocol

<!-- SID:E01 -->

Probability evaluation included AUROC, AUPRC, Brier score, binary log
loss, calibration intercept, calibration slope, and absolute deviations
of the two calibration coefficients from their ideal values zero and
one. Skill scores used a prevalence-only predictor on the same
evaluation set as a reference. Conformal evaluation included marginal
coverage, class-specific and worst-class coverage, worst-class
undercoverage, absolute marginal-coverage error, mean set size, and
empty-, singleton-, and doubleton-set fractions.

<!-- SID:E02 -->

H2 paired each positive-budget target-fitted probability row with the
same method's source-fitted budget-zero row within task, split, base
model, model seed, adaptation seed, budget, and sample hash.
Benefit-oriented differences were defined so that positive values
indicated improvement. The prespecified material thresholds were an
absolute log-loss reduction of at least 0.002, a relative log-loss
reduction of at least 1%, and a Brier-skill gain of at least 0.005.
Availability and algorithm-estimability rates used all attempts. Benefit
and harm rates used only successful finite pairs; `not_available` and
`not_estimable` were never interpreted as absence of harm.

<!-- SID:E03 -->

H3 constructed complete budget curves only for the five tasks supporting
all budgets through 500. Each cell retained absolute budget, target-pool
fraction, and observed positive and negative label counts. Repeated
acquisition effects were first collapsed within task. Cross-task
summaries then used equal task weighting, with equal source-family
weighting as a secondary view so that the two ACS tasks did not
implicitly double the ACS family's contribution. Leave-one-task-out
(LOTO) and leave-one-source-family-out (LOFO) analyses were recomputed
from the paired H2 effects and interpreted as deletion-stability checks,
not independent replications. College remained a separate 25–100 stress
analysis. Target-pool fractions were task-specific descriptors of the
tested absolute budgets; the protocol did not evaluate a shared grid of
relative target-pool fractions.

<!-- SID:E04 -->

H4 directly paired budget-zero results on `source_id_test` and
`target_final_test`. Raw differences were target-final minus source-ID.
The primary probability comparison used uncalibrated base-model scores;
source-fitted calibrators were supplementary. H4 was descriptive: it
assessed whether discrimination and reliability metrics could move
differently under the evaluated domain shifts, without causal
attribution or a claim that AUROC is independent of, or irrelevant to,
reliability.

<!-- SID:E05 -->

H5 compared Mondrian with standard conformal prediction within each
matched attempt. Metric differences were defined only when both methods
completed successfully and the resulting contrast was finite. Within
each task–model–budget cell, we took the median of the jointly
successful paired differences, retaining successful Mondrian outputs
with infinite thresholds in the primary conditional summary; we then
averaged the 10 cell medians equally at each budget. A sensitivity
analysis further restricted pairs to finite Mondrian thresholds and
averaged the available cell medians. Mondrian estimability and
finite-threshold rates were calculated over available attempts; all
primary-set configurations at the five tested positive budgets were
available. Positive worst-class-coverage differences represented higher
observed Mondrian coverage, whereas positive mean-set-size differences
represented larger prediction sets. Non-estimable attempts remained in
the feasibility accounting and were not assigned a zero metric
difference. Coverage, feasibility, and set-size effects were reported
jointly; no single difference was used as an overall method ranking.

<!-- SID:E06 -->

Descriptive summaries reported task/model-stratified effects,
conditional event rates, eligibility rates, and deletion stability.
These quantities represent descriptive heterogeneity rather than
run-level confidence intervals. Task was the primary aggregation unit
and source family the secondary unit. No run-level bootstrap, global
run-level $`p`$-value, or row-level pseudo-replication was used. H1
retained predicted class-support probability, empirical class-support
rate, empirical algorithm-estimability rate, and empirical
finite-threshold rate as distinct quantities. Exact finite-population
support under the observed pool composition was a retrospective
validation only; deployment-style planning used the predeclared
prevalence interval $`[0.01,0.99]`$. Run-level binary-event MAE and RMSE
were not interpreted as probability-versus-rate calibration error.

<!-- SID:E07 -->

Completed simulations were used only as mechanism probes. The
probability grid contained seven mechanisms—reference, mild and severe
intercept shifts, slope shift, nonlinear shift, subgroup shift, and
compressed scores—while the conformal grid contained reference,
overconfidence, class-asymmetric, and rare-class mechanisms. The grids
crossed target prevalences 0.01, 0.05, 0.20, and 0.50 with budgets
25–500 and 200 fixed repetitions per cell. Simulation rows were
summarized separately from natural-shift rows and were not treated as
external validation. In particular, the simulation-specific sigmoid
support convention was not substituted for the natural-shift
implementation's requirement that both outcome classes be represented.

<!-- SID:E08 -->

All reported aggregates were deterministic transformations of the
recorded machine-readable tables. For H2, model-specific task-equal
summaries were averaged equally over logistic regression and CPU
XGBoost; no run pooling was used. For H5, the primary conditional
summary at each budget was an equal mean over the 10 primary task–model
cells after taking the within-cell median over jointly successful paired
metric differences. The finite-threshold sensitivity summary averaged
the available cell medians after additionally requiring a finite
Mondrian threshold. Figures display those aggregates or the underlying
evaluated task cells, and every panel has a CSV source-data file
specifying the selected rows and transformation. This separation between
experiment output and presentation output permits an audit of rounding,
filtering, sign conventions, and denominators without reopening model
execution.

## 7. Results

### 7.1 Experimental completion and eligibility

<!-- SID:R71-01 -->

All 54 prespecified model configurations completed, covering six tasks,
three split seeds, logistic regression, and the two CPU XGBoost
model-seed strata. The experiment inventory contained no technical
failure row. Probability results comprised 98,796 successful, 7,200
`not_available`, and 2,544 `not_estimable` rows; conformal results
comprised 96,600 successful, 7,200 `not_available`, and 4,632
`not_estimable` rows. The threshold result family was a view of the same
conformal attempts and was not added again when counting failures.

<!-- SID:R71-02 -->

The combined method-level `not_estimable` records therefore contained
7,176 rows. Of the probability rows, 114 intercept-only MLE attempts and
114 sigmoid attempts lacked both classes, while 2,316 isotonic attempts
lacked its required per-class support. The 4,632 conformal rows were
Mondrian attempts with fewer than five labels in at least one class. All
`not_available` positive-budget rows arose from prespecified task–budget
combinations rather than technical termination; in particular, College
budgets 250 and 500 remained explicit instead of being dropped.

<!-- SID:R71-03 -->

The auxiliary H1 audit separated basic class support, method
eligibility, and finite thresholds. Across the six tasks, empirical
support for at least one label per class increased from 98.1% at budget
25 to 99.8% at 50 and 100% from 100 onward. Support for at least five
labels per class was 69.2%, 89.4%, 98.6%, 100%, and 100% across budgets
25, 50, 100, 250, and 500. Method-specific eligibility remained a
separately reported operational coordinate rather than an inferred
performance outcome.

<!-- SID:R71-04 -->

Finite Mondrian thresholds were more restrictive at small budgets. The
corrected rate-level audit gave a predicted mean finite-threshold rate
of 0.779157 and an empirical mean of 0.779643. Empirical rates across
the five positive budgets were 0.3306, 0.7317, 0.9100, 0.9993, and
1.0000. These values describe the realized six-task design and do not
convert known target composition into prospective deployment
information. Detailed rate-level validation is reported in the
supplement.

### 7.2 Complexity, budget, benefit and harm

<!-- SID:R72-01 -->

Figure 2 and Table 3 show the primary five-task H2 results, averaged
equally over the two prespecified base-model strata after task-level
aggregation. Positive change denotes lower target-final log loss
relative to the same update fitted on the source probability-calibration
role at budget zero. Material harm denotes a paired log-loss change of
at most −0.002 and is calculated only among successful finite pairs.
Eligibility is shown separately, so a method that could not be fitted
was not counted as harmless.

![Complexity–budget response for the four probability updates. Panel A
shows the task-equal mean, task–model interquartile range, and
minimum-to-maximum range on a symmetric logarithmic scale; these
descriptive spreads are not confidence intervals. The dashed line marks
the +0.002 material-benefit threshold. Panel B shows conditional
material-harm rates.](../figures/figure2_complexity_budget.png){#fig:h2 width=100%}

<!-- SID:R72-02 -->

Intercept-only MLE moved from a near-zero mean benefit of −0.000103 at
budget 25 to +0.004040 at 50, +0.006711 at 100, and +0.008803 at 500.
Its material-harm rate nevertheless remained 43.1% at budget 25 and
30.4% at 50 before falling to 15.0%, 4.2%, and 1.0% at budgets 100, 250,
and 500. Thus, the earlier average recovery of this low-dimensional
update did not make the smallest budget risk-free.

<!-- SID:R72-03 -->

Jeffreys smoothing was estimable in 100% of primary-set attempts at
every positive budget. Its mean benefit changed from +0.000342 at budget
25 to +0.008680 at 500, while its harm rate changed from 43.6% to 1.0%.
The corresponding curves closely tracked unsmoothed intercept-only MLE.
The recorded evidence therefore supports a low-budget availability
advantage, but not a clear performance advantage over the unsmoothed
intercept update.

<!-- SID:R72-04 -->

Sigmoid calibration's mean log-loss benefit was −0.025451 at budget 25
and −0.006135 at 50. The mean became positive at budget 100 (+0.002534),
one tested budget step later than intercept MLE, but 33.5% of successful
finite pairs still met the material-harm definition. At budget 250 the
mean benefit was +0.006874 and harm fell to 8.9%; at 500 these values
were +0.008486 and 1.7%. The average sign at budget 100 therefore did
not by itself mark a low-risk point.

<!-- SID:R72-05 -->

Isotonic calibration became increasingly estimable—from 64.5% at budget
25 to 87.3% at 50, 98.3% at 100, and 100% at 250 and 500—but its
primary-set mean log-loss benefit remained negative: −0.4218, −0.2595,
−0.1365, −0.0477, and −0.01722. Its conditional material-harm rate
declined from 99.7% to 73.2% across the same grid. Persistent harm at
the upper tested budget was therefore not explained by ineligibility
alone. This statement is confined to the evaluated natural-shift grid
and does not characterize isotonic behavior in other settings.

**Table 3. Common-five probability-update effects, material-benefit
heterogeneity, and deletion stability. Mean change is task-equal and
then equal over the two model strata.**

| Method | Budget | Mean Δ log loss | Material harm | Estimable | Material-positive task–model cells / 10 | Worst task–model median | LOTO expected direction retained / 10 |
|----|----|----|----|----|----|----|----|
| Intercept MLE | 25 | -0.0001 | 43.1% | 97.7% | 4 | -0.0120 | 5 |
| Intercept MLE | 50 | +0.0040 | 30.4% | 99.8% | 5 | -0.0050 | 9 |
| Intercept MLE | 100 | +0.0067 | 15.0% | 100.0% | 6 | -0.0026 | 10 |
| Intercept MLE | 250 | +0.0082 | 4.2% | 100.0% | 6 | -0.0010 | 10 |
| Intercept MLE | 500 | +0.0088 | 1.0% | 100.0% | 7 | -0.0006 | 10 |
| Jeffreys intercept | 25 | +0.0003 | 43.6% | 100.0% | 5 | -0.0116 | 6 |
| Jeffreys intercept | 50 | +0.0044 | 29.4% | 100.0% | 5 | -0.0050 | 10 |
| Jeffreys intercept | 100 | +0.0067 | 14.3% | 100.0% | 6 | -0.0026 | 10 |
| Jeffreys intercept | 250 | +0.0081 | 4.3% | 100.0% | 7 | -0.0010 | 10 |
| Jeffreys intercept | 500 | +0.0087 | 1.0% | 100.0% | 7 | -0.0006 | 10 |
| Sigmoid | 25 | -0.0255 | 74.0% | 97.7% | 0 | -0.0400 | 10 |
| Sigmoid | 50 | -0.0061 | 57.3% | 99.8% | 2 | -0.0149 | 10 |
| Sigmoid | 100 | +0.0025 | 33.5% | 100.0% | 4 | -0.0052 | 8 |
| Sigmoid | 250 | +0.0069 | 8.9% | 100.0% | 5 | -0.0012 | 10 |
| Sigmoid | 500 | +0.0085 | 1.7% | 100.0% | 6 | -0.0004 | 10 |
| Isotonic | 25 | -0.4218 | 99.7% | 64.5% | 0 | -0.5634 | 10 |
| Isotonic | 50 | -0.2595 | 98.3% | 87.3% | 0 | -0.3456 | 10 |
| Isotonic | 100 | -0.1365 | 95.1% | 98.3% | 0 | -0.1733 | 10 |
| Isotonic | 250 | -0.0477 | 86.5% | 100.0% | 0 | -0.0702 | 10 |
| Isotonic | 500 | -0.0172 | 73.2% | 100.0% | 2 | -0.0342 | 10 |

### 7.3 No common tested absolute budget within the evaluated grid

<!-- SID:R73-01 -->

No evaluated method–budget combination achieved the prespecified +0.002
log-loss improvement in all 10 common task–model cells. Figure 3 makes
the source of this result visible: task-specific median effects differed
in magnitude and, at several budgets, in sign within both logistic
regression and CPU XGBoost. Intercept updates generally entered positive
regions earlier than sigmoid, but individual tasks did not share one
boundary. Isotonic remained negative in most cells even when its
task-specific curve improved with budget.

![Task-, model-, method-, and budget-specific median log-loss benefit
across the primary five-task set. Each cell is a task–model summary;
College is excluded. The symmetric logarithmic color scale retains
small changes around zero while showing large isotonic harms; open
circles mark absolute effects below
0.002.](../figures/figure3_common5_heterogeneity.png){#fig:h3 width=100%}

<!-- SID:R73-02 -->

Absolute label counts also represented different fractions of the target
adaptation pools. At budget 100, the split-level H3 curves gave
fractions of 0.087% for BRFSS Diabetes, 0.296% for ACS Income, 0.663%
for Diabetes Readmission, 1.136% for ACS Food Stamps, and 1.970% for
NHANES Lead—a 22.6-fold range between the smallest and largest
common-task fractions. These fractions describe how the tested absolute
budgets mapped to each task; the study did not evaluate a common
relative-budget grid and therefore does not identify a universal
target-pool-fraction threshold.

<!-- SID:R73-03 -->

The principal directions were stable to the prespecified deletion
checks: 169 of 170 LOTO checks and all 136 LOFO checks retained their
expected direction. The single LOTO reversal had magnitude approximately
−0.000028. These analyses show that the observed pattern was not driven
solely by one task or one source family, but they are deletion-stability
descriptions within the observed sample, not additional population
replications.

<!-- SID:R73-04 -->

College Scorecard remained separate. Its registry adaptation capacity
was 107, while the executed split-level H3 curve placed budget 100 at
approximately 96.2% of the available adaptation pool. Budgets 250 and
500 were `not_available`. The lower-budget effects also showed a strong
logistic-regression versus XGBoost interaction, reinforcing the decision
not to pool this small-domain stress task with the complete primary-set
curves.

### 7.4 Class-conditional reliability and set-size cost

<!-- SID:R74-01 -->

Among jointly successful standard–Mondrian pairs across the primary
five-task set, two base models, five positive budgets, and the primary
uncalibrated score source, the median difference in worst-class coverage
was positive in each of the 50 task–model–budget cells. Thirty-eight
cell medians were at least 0.10, and the direction agreed between the
two model strata in all 25 task–budget combinations. These metric
comparisons are conditional on both methods producing successful
results; estimability and finite-threshold rates are reported separately
over available attempts.

![Paired Mondrian-minus-standard results. Panels A and B compare the
joint-success conditional summary, which retains infinite Mondrian
thresholds, with a sensitivity analysis restricted to jointly
successful pairs with finite Mondrian thresholds. Panel C reports
Mondrian estimability and finite-threshold rates across available
attempts.](../figures/figure4_standard_mondrian_tradeoff.png){#fig:h5 width=100%}

<!-- SID:R74-02 -->

The joint-success conditional mean worst-class-coverage difference rose
from +0.1285 at budget 25 to +0.2876 at 50, +0.3420 at 100, +0.3505 at
250, and +0.3541 at 500. Mean set size in the same conditional summary
simultaneously increased by +0.3280, +0.3469, +0.2843, +0.2069, and
+0.1975. The associated doubleton-fraction changes were +0.3278,
+0.3465, +0.2776, +0.2030, and +0.1923, showing that most of the binary
set expansion came from full two-label outputs rather than a change
between empty and singleton sets.

<!-- SID:R74-03 -->

Feasibility changed sharply with budget. Across available attempts,
Mondrian estimability was 64.5%, 87.3%, 98.3%, 100%, and 100% across
budgets 25–500; finite-threshold rates were only 30.1% and 68.2% at 25
and 50, then 89.2%, 99.9%, and 100%. In the finite-threshold-only
successful subset at budget 25, the mean worst-class gain was
approximately +0.0275 and the mean set-size change approximately
+0.0013. This descriptive subset represented only four tasks and eight
task–model cells because NHANES had no finite row at that budget. The
larger joint-success conditional budget-25 averages were therefore
strongly influenced by infinite thresholds and full-label sets. This
low-budget distinction is expanded in the supplement.

<!-- SID:R74-04 -->

Marginal coverage and absolute marginal-coverage error did not improve
uniformly. Mean raw Mondrian-minus-Standard marginal-coverage changes
were −0.0085, +0.0083, +0.0084, +0.0038, and +0.0029 over the five
budgets, while absolute-error differences changed sign. The H5 result is
thus a class-conditional coverage–feasibility–set-cost trade-off, not an
overall ordering of Standard and Mondrian.

**Table 4. Paired Mondrian-minus-standard conditional metric
differences, feasibility, and finite-threshold sensitivity.**

| Budget | Joint-success Δ worst-class coverage | Joint-success, finite-threshold Δ worst-class coverage | Joint-success Δ mean set size | Joint-success, finite-threshold Δ mean set size | Finite task–model cells | Mondrian estimable (available attempts) | Finite threshold (available attempts) |
|----|----|----|----|----|----|----|----|
| 25 | +0.1285 | +0.0275 | +0.3280 | +0.0013 | 8/10 | 64.5% | 30.1% |
| 50 | +0.2876 | +0.2044 | +0.3469 | +0.1984 | 10/10 | 87.3% | 68.2% |
| 100 | +0.3420 | +0.3167 | +0.2843 | +0.2227 | 10/10 | 98.3% | 89.2% |
| 250 | +0.3505 | +0.3505 | +0.2069 | +0.2068 | 10/10 | 100.0% | 99.9% |
| 500 | +0.3541 | +0.3541 | +0.1975 | +0.1975 | 10/10 | 100.0% | 100.0% |

Metric-difference columns condition on joint success; the primary
joint-success columns retain successful Mondrian outputs with infinite
thresholds. Feasibility columns use available-attempt denominators, and
non-estimable attempts are not assigned zero differences.

### 7.5 Auxiliary discrimination–reliability diagnostic

<!-- SID:R75-01 -->

The primary H4 diagnostic contained 54 strictly paired, budget-zero
source-ID and target-final comparisons using uncalibrated base-model
probabilities. Among the 17 pairs with strict
$`|\Delta\mathrm{AUROC}|<0.01`$, Brier score and log loss both worsened
in 17 of 17; their median raw increases were +0.0271 and +0.0717. At the
strict 0.02 threshold, both scores worsened in all 32 of 32 pairs, with
median increases +0.0182 and +0.0467. These are task–split–model-design
pairs, not independent task replications; the 17-pair threshold set is
nested within the 32-pair set. Thus, small AUROC changes did not obviate
direct evaluation of Brier score and log loss in these comparisons.

![Source-to-target changes for the 54 primary uncalibrated H4 pairs. Raw
horizontal differences are target-domain test minus source-domain test
AUROC; positive vertical differences are worse Brier score or log loss.
Dashed lines mark absolute AUROC changes of 0.01; star markers identify
the separate College stress task, and the horizontal axis uses a
symmetric-log transform.](../figures/figure5_auroc_reliability.png){#fig:h4 width=100%}

<!-- SID:R75-02 -->

Across all 54 design pairs, AUROC changes were strongly negatively
associated with Brier-score and log-loss changes: Pearson correlations
were −0.908 and −0.895; the corresponding Spearman correlations were
−0.762 and −0.772. Because multiple pairs arose from the same tasks,
these descriptive correlations are not population-level estimates. AUPRC
and calibration-coefficient diagnostics also did not all move in one
direction. AUROC remains a discrimination measure; the bounded result is
that it did not replace direct probability-reliability evaluation.

### 7.6 Mechanism-probe simulations

<!-- SID:R76-01 -->

The completed probability simulations qualitatively reproduced the
small-budget sensitivity of more complex updates, and the conformal
simulations reproduced a class-asymmetry mechanism in which classwise
coverage could require larger sets. They did not reproduce all
natural-shift behavior. Simulated isotonic mean log-loss benefit was
approximately −0.410 and −0.172 at budgets 25 and 50, then became
positive at about +0.058, +0.197, and +0.240 at budgets 100, 250, and
500. In contrast, the primary-set natural-shift mean remained negative
through 500.

<!-- SID:R76-02 -->

The archived analysis review placed sigmoid's simulated reduction in
harm near budget 100. In the natural-shift analysis, however, 33.5% of
its successful finite pairs at budget 100 still met the material-harm
criterion, and the rate fell below 10% only at budget 250 (8.9%). The
simulation grid therefore supports a broad complexity–sample-size
mechanism while disagreeing on the budget location and high-budget
isotonic pattern. Conformal finite-threshold rates in the simulation
summary were conditional on estimable successes, so they were not
directly subtracted from the available-attempt natural-shift rates. The
simulations did not contain the source-domain/target-domain test pairs
required for H4.

## 8. Discussion

<!-- SID:D01 -->

The central empirical result is that update complexity and target-label
budget acted jointly, but the ordering of mean benefits across tested
budgets did not yield one budget that satisfied the prespecified
improvement criterion across task–model–method cells. Intercept updates
were the lower-dimensional anchors, and their mean benefit became
positive at a lower tested budget; sigmoid generally required more
labels to reduce harm; and isotonic retained substantial log-loss harm
within the tested natural-shift range. Yet even the intercept methods
had material harm in more than two fifths of successful budget-25 pairs.
A mean positive effect is therefore insufficient for a reliability
decision when the distribution of paired harms remains wide.

<!-- SID:D02 -->

This pattern is consistent with established work on calibration sample
size and staged prediction-model updating ([Niculescu-Mizil and Caruana
2005](#ref-niculescu2005goodprob); [Steyerberg et al.
2004](#ref-steyerberg2004updating); [Vergouwe et al.
2017](#ref-vergouwe2017closed); [Ojeda et al.
2023](#ref-ojeda2023comparison); [Sun et al.
2023](#ref-sun2023minimumrisk)). The contribution here is not the
qualitative mechanism. It is the joint measurement of benefit, material
harm, availability, target-pool fraction, and deletion stability under
one label-blind random protocol spanning multiple natural tabular shifts
and two base predictors. This protocol-level view exposes a decision
problem that a method-average score can hide: whether a proposed update
can be fitted, how frequently it produces a material loss, and whether
the same nominal budget has comparable meaning across targets.

<!-- SID:D03 -->

H3 sharpens that point. A budget of 100 labels ranged from a very small
fraction of a large BRFSS adaptation pool to almost the entire executed
College pool. Relative budget alone also cannot encode minority-class
support, score structure, base-model calibration, or update dimension.
Because a common relative-budget grid was not tested, the observed pool
fractions should be read as task-specific descriptors rather than as
evidence for or against a universal fraction threshold. The appropriate
interpretation is conditional: for the evaluated tasks, the evidence can
identify regions with lower observed risk, but it cannot turn budget
250—or any other tested value—into a general operating rule. The LOTO
and LOFO checks indicate that no single task or source family alone
determined the overall direction; however, these remain descriptive
sensitivity analyses within the observed task set.

<!-- SID:D04 -->

The Jeffreys comparison illustrates why availability and performance
should be separated. Smoothing removed the requirement that both outcome
classes be represented and produced complete low-budget estimability,
but its benefit and harm curves stayed close to unsmoothed intercept
MLE. For a user who values a finite update in a single-class sample,
that availability is operationally relevant. It is not evidence that
smoothing improves predictive performance in general. Similarly,
isotonic's rising estimability did not make its fitted updates
beneficial on average in the evaluated natural tasks. Eligibility is a
prerequisite for a performance comparison, not a performance result.

<!-- SID:D05 -->

The H5 findings add a separate class-conditional decision layer. Among
jointly successful pairs, Mondrian increased observed worst-class
coverage in every primary task–model–budget cell, but low-budget
thresholds were often infinite and prediction sets expanded. In a binary
problem, that expansion was mainly an increase in doubletons, which
express uncertainty by returning both labels. Whether this is acceptable
depends on the decision context: an application that prioritizes the
class with lower observed coverage may accept a larger set, whereas an
application requiring actionable singleton outputs may not. Recent
class-conditional conformal research already establishes this
coverage–efficiency tension and develops other ways to navigate it
([Ding et al. 2023](#ref-ding2023classconditional),
[2026](#ref-ding2026longtailed); [Shi et al.
2024](#ref-shi2024classwise); [Liu et al. 2026](#ref-liu2026longtail);
[Papaioannou et al. 2026](#ref-papaioannou2026robust); [Althani
2026](#ref-althani2026extreme)). Our result contributes the paired
budget response under the same natural-tabular adaptation samples, not a
claim that the tension itself was previously unknown.

<!-- SID:D06 -->

Retaining unsuccessful and non-estimable attempts changes the
interpretation of both probability and conformal results. A method with
attractive conditional performance but low eligibility cannot be
compared directly with a method that produces a result in every attempt
unless the availability difference is visible. The explicit
`not_available`, `not_estimable`, infinite-threshold, and `failed`
states prevent an apparently clean curve from being created by deleting
difficult samples. They also separate different remedies: acquiring more
total labels addresses some support failures, smoothing changes an
intercept fit's eligibility rule, and finite classwise conformal
thresholds require adequate support in each outcome class.

<!-- SID:D07 -->

The H4 diagnostic reinforces the need for a multi-metric audit without
devaluing discrimination. AUROC answers a ranking question, whereas
Brier score and log loss assess probability quality. The descriptive
associations between their source-to-target changes were strong, but the
near-zero-AUROC subsets demonstrate that a small ranking change did not
rule out a meaningful proper-score deterioration. This distinction is
established in the calibration and model-shift literature ([Davis et al.
2017](#ref-davis2017aki); [Levy et al. 2022](#ref-levy2022autoupdating);
[Guo et al. 2023](#ref-guo2023ehrshift); [Ovadia et al.
2019](#ref-ovadia2019trust); [Van Calster et al.
2019](#ref-vancalster2019achilles)); the present counts are auxiliary
cross-task evidence under the prespecified protocol.

<!-- SID:D08 -->

Simulation provides a final caution against turning a plausible
mechanism into a complete account of the natural results. The simulated
complexity ordering and conformal set expansion were directionally
informative, but simulated isotonic calibration improved at high budgets
while the natural-shift aggregate did not. Natural shifts can combine
prevalence, covariate, conditional, and score-geometry changes that are
simplified in a mechanistic grid. The disagreement is scientifically
useful because it bounds the interpretation: the simulation can
illustrate why small samples may destabilize a flexible update, but it
cannot override the empirical behavior of the evaluated natural tasks.

<!-- SID:D09 -->

Taken together, the results suggest an audit sequence for similar
bounded studies. Investigators can define entity-disjoint source and
target roles, acquire labels without outcome-aware selection, record
class support and method eligibility, compare each update with its own
source-fitted reference, and examine benefit and material harm before
selecting a budget. If class-conditional prediction sets are considered,
finite thresholds and set composition should accompany coverage. This
sequence is a reporting and decision discipline derived from the
evaluated protocol, not a claim that the tested methods or budgets
transfer unchanged to another application.

<!-- SID:D10 -->

The results also argue for separating a budget recommendation into at
least three questions. The sampling question asks how much of the target
pool is labeled and which class counts are likely under a label-blind
draw. The estimation question asks whether the chosen updater or
classwise threshold can be fitted under that support. The performance
question asks whether successful fits improve a proper score or coverage
objective often enough, and at an acceptable cost. A method can pass one
question and fail another: Jeffreys smoothing can be available without
outperforming its unsmoothed analogue, isotonic can become estimable
while remaining harmful on average, and Mondrian can increase
worst-class coverage while returning more doubletons. Reporting these
questions separately makes task-specific trade-offs visible and reduces
the temptation to treat a positive mean or a nominal label count as a
complete decision.

## 9. Limitations

<!-- SID:L01 -->

The empirical sample is limited to six public binary tabular tasks from
five source families. The primary pooled results use only five tasks
from four families because College lacks the two upper budgets. Two base
predictors—logistic regression and one CPU XGBoost configuration
family—cannot represent the behavior of other boosting systems, random
forests, support-vector machines, neural tabular models, foundation
models, ensembles, or models trained with different tuning objectives.
The study does not cover multiclass outcomes, regression targets,
structured outputs, or repeated-event prediction.

<!-- SID:L02 -->

The update set is also narrow. Probability analyses are limited to the
prespecified uncalibrated, intercept-only MLE, Jeffreys-smoothed
intercept, sigmoid, and isotonic implementations. Conformal analyses
compare Standard and outcome-class Mondrian methods with two fixed score
sources and nominal coverage 0.90. The study does not evaluate beta or
temperature calibration, arbitrary nonparametric updates, full model
refitting, representation adaptation, weighted or online conformal
methods, or hierarchical classwise fallbacks. H6 was outside the final
scope, so the evidence says nothing about outcome-aware, active, or
alternative label-acquisition policies.

<!-- SID:L03 -->

Budgets are restricted to 25, 50, 100, 250, and 500 label-blind random
target labels, with budget zero as a source-fitted comparison reference.
The discrete grid does not identify an exact transition between tested
values, and there is no evidence beyond 500. Absolute counts and
target-pool fractions do not fully determine realized per-class support.
College is a separate stress case with only the lower budgets, and its
high sampling fraction should not be extrapolated to larger tasks.

<!-- SID:L04 -->

The source–target domains are public observational partitions defined by
geography, survey group, poverty, admission source, or institutional
control. They represent natural shifts but are not randomized
interventions. The analysis cannot identify why a metric changed,
isolate a causal shift component, establish class or group fairness
beyond the reported outcome-class coverage, or certify safe deployment.
Results are also bounded to the countries, periods, inclusion rules,
feature definitions, class prevalences, and public-data measurement
processes represented by these six tasks.

<!-- SID:L05 -->

Repeated splits, model seeds, and adaptation samples improve description
of the fixed design but do not create independent task replications.
XGBoost seed strata are model-design repetitions, and H1 model rows
duplicate the same label-sampling design calculation. Task-equal and
family-equal summaries, together with LOTO and LOFO analyses, describe
sensitivity within this sample. They do not establish a population
distribution over tabular shifts. Harm rates are conditional on
successful finite pairs and must be read alongside the separate
eligibility rates.

<!-- SID:L06 -->

The conformal comparison is empirical under the prespecified
source–target construction. Natural shift can violate exchangeability
assumptions that underlie standard finite-sample conformal statements;
the paper reports realized target coverage rather than asserting a
distribution-free target-domain property. At low budgets, infinite
Mondrian thresholds can produce full binary sets, so higher worst-class
coverage need not mean more informative prediction. Marginal coverage
and absolute coverage error also did not improve consistently.

<!-- SID:L07 -->

The H4 analysis is a diagnostic association. It does not show that AUROC
change causes proper-score change, that the metrics are independent, or
that discrimination lacks value. The simulations are mechanism probes
rather than held-out natural tasks. Their restricted data-generating
mechanisms do not reproduce every feature of real domain shift, and the
high-budget isotonic disagreement is a material limitation rather than a
result to average away. The simulation grid also lacks the
source-ID/target-final structure needed to examine H4.

<!-- SID:L08 -->

Finally, literature positioning is bounded to the completed search and
verification record. No complete duplicate of the joint protocol was
identified in that record, but inaccessible or later work may alter this
assessment. No claim in this article relies on literature that was
available only as unverified metadata.

## 10. Conclusion

<!-- SID:C01 -->

Within the primary five-task set, two tested base models, four
probability updates, and label-blind random target-label budgets of
25–500, calibrator complexity and budget jointly shaped mean benefit and
the observed material-harm rate among successful finite pairs. The mean
benefit of intercept updating became positive at a lower tested budget
than that of sigmoid, while isotonic retained negative mean log-loss
benefit through the upper tested budget. No tested absolute budget met
the prespecified improvement condition across the evaluated
task–model–calibrator grid. Because the design did not test a common
relative-budget grid, it does not identify a universal fraction
threshold.

<!-- SID:C02 -->

The class-conditional analysis reached a complementary conclusion. Among
jointly successful pairs, the median Mondrian-minus-standard
worst-class-coverage difference was positive in all 50 primary
task–model–budget cells, but estimability, finite-threshold rates, and
prediction-set expansion varied strongly with budget. The observed
coverage differences were therefore inseparable from feasibility and
binary set cost. The bounded practical implication is to evaluate
method, task, model, class support, and budget together, retain
unsuccessful attempts in the record, and report probability and
set-valued reliability on their own metrics.

## Data and Code Availability

<!-- SID:AV01 -->

The study used public secondary datasets governed by the licenses and
terms recorded in the repository's data-license registry; official
source and access details are listed in Supplementary Table S8. The
public release contains manuscript artifact sources, figure- and
table-level machine-readable source data, analysis and
artifact-generation code, environment specifications, checksums, and a
documented workflow for one representative end-to-end ACS model shard.
An optional workflow re-executes the 54 prespecified model shards from
prepared, audited Parquet inputs; it does not reconstruct every source
dataset or regenerate the complete intermediate analysis tree. The
release does not redistribute raw third-party datasets. Data-acquisition
instructions identify official sources and applicable provider terms.
Author-created software is licensed under the MIT License. The reviewed
release is available in the [public GitHub
repository](https://github.com/ChenDH9/reliable-tabular-shift) at tag
`v1.0.0` and is archived under the version-specific DOI
[10.5281/zenodo.21354982](https://doi.org/10.5281/zenodo.21354982).

## Funding

The author received no specific funding for this work.

## Competing Interests

The author declares that there are no competing interests.

## Author Contributions / CRediT

Dinghao Chen: Conceptualization, Data curation, Formal analysis,
Investigation, Methodology, Software, Validation, Visualization, Writing
- original draft, and Writing - review & editing. The author accepts full
responsibility for the work.

## Acknowledgments

The author thanks the College of Big Data, Baoshan University, for
providing access to the computing facilities used in this study.

## Declaration of Generative AI-Assisted Work

<!-- SID:AI01 -->

OpenAI Codex (GPT-5; accessed 11-14 July 2026) was used solely for
manuscript formatting, language refinement, file-consistency checks, and
pre-submission review to improve clarity, consistency, and compliance
with submission requirements. It was not used to formulate the research
question, design the study, conduct experiments, analyze data, interpret
results, formulate scientific claims, or draw conclusions; these
activities were completed independently by the author. The author
reviewed all AI-assisted changes, confirmed the originality and accuracy
of the manuscript, verified all references, checked the tool's terms of
use and suitability for publication, retained the original and revised
versions, and accepts full responsibility for the integrity of the
content. Generative AI was not used to create or modify research data or
the scientific features of any figure.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-althani2026extreme" class="csl-entry">

Althani, Bashair. 2026. “Class-Conditional Conformal Prediction for
Reliable Anomaly Detection Under Extreme Class Imbalance.” *Machine
Learning and Knowledge Extraction* 8 (7): 190.
<https://doi.org/10.3390/make8070190>.

</div>

<div id="ref-angelopoulos2023gentle" class="csl-entry">

Angelopoulos, Anastasios N., and Stephen Bates. 2023. “Conformal
Prediction: A Gentle Introduction.” *Foundations and Trends in Machine
Learning* 16 (4): 494–591. <https://doi.org/10.1561/2200000101>.

</div>

<div id="ref-cdcbrfss2015" class="csl-entry">

Centers for Disease Control and Prevention. 2015. *2015 BRFSS Survey
Data and Documentation*. National Center for Chronic Disease Prevention;
Health Promotion, Division of Population Health.
<https://www.cdc.gov/brfss/annual_data/annual_2015.html>.

</div>

<div id="ref-cdcbrfss2017" class="csl-entry">

Centers for Disease Control and Prevention. 2017. *2017 BRFSS Survey
Data and Documentation*. National Center for Chronic Disease Prevention;
Health Promotion, Division of Population Health.
<https://www.cdc.gov/brfss/annual_data/annual_2017.html>.

</div>

<div id="ref-cdcbrfss2019" class="csl-entry">

Centers for Disease Control and Prevention. 2019. *2019 BRFSS Survey
Data and Documentation*. National Center for Chronic Disease Prevention;
Health Promotion, Division of Population Health.
<https://www.cdc.gov/brfss/annual_data/annual_2019.html>.

</div>

<div id="ref-cdcbrfss2021" class="csl-entry">

Centers for Disease Control and Prevention. 2021. *2021 BRFSS Survey
Data and Documentation*. National Center for Chronic Disease Prevention;
Health Promotion, Division of Population Health.
<https://www.cdc.gov/brfss/annual_data/annual_2021.html>.

</div>

<div id="ref-clore2014diabetes130" class="csl-entry">

Clore, John, Krzysztof Cios, Jonathan DeShazo, and Beata Strack. 2014.
*Diabetes 130-US Hospitals for Years 1999–2008*. UCI Machine Learning
Repository. <https://doi.org/10.24432/C5230J>.

</div>

<div id="ref-davis2020protocols" class="csl-entry">

Davis, Sharon E., Robert A. Greevy, Thomas A. Lasko, Colin G. Walsh, and
Michael E. Matheny. 2020. “Comparison of Prediction Model Performance
Updating Protocols: Using a Data-Driven Testing Procedure to Guide
Updating.” *AMIA Annual Symposium Proceedings* 2019: 1002–10.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC7153129/>.

</div>

<div id="ref-davis2017aki" class="csl-entry">

Davis, Sharon E., Thomas A. Lasko, Guanhua Chen, Edward D. Siew, and
Michael E. Matheny. 2017. “Calibration Drift in Regression and Machine
Learning Models for Acute Kidney Injury.” *Journal of the American
Medical Informatics Association* 24 (6): 1052–61.
<https://doi.org/10.1093/jamia/ocx030>.

</div>

<div id="ref-ding2021folktables" class="csl-entry">

Ding, Frances, Moritz Hardt, John Miller, and Ludwig Schmidt. 2021.
“Retiring Adult: New Datasets for Fair Machine Learning.” *Advances in
Neural Information Processing Systems* 34: 6478–90.
<https://proceedings.neurips.cc/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html>.

</div>

<div id="ref-ding2023classconditional" class="csl-entry">

Ding, Tiffany, Anastasios Angelopoulos, Stephen Bates, Michael I.
Jordan, and Ryan J. Tibshirani. 2023. “Class-Conditional Conformal
Prediction with Many Classes.” *Advances in Neural Information
Processing Systems* 36: 64555–76.
<https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html>.

</div>

<div id="ref-ding2026longtailed" class="csl-entry">

Ding, Tiffany, Jean-Baptiste Fermanian, and Joseph Salmon. 2026.
“Conformal Prediction for Long-Tailed Classification.” *International
Conference on Learning Representations*.
<https://openreview.net/forum?id=8L83ZbFDjk>.

</div>

<div id="ref-fisch2021fewshot" class="csl-entry">

Fisch, Adam, Tal Schuster, Tommi Jaakkola, and Regina Barzilay. 2021.
“Few-Shot Conformal Prediction with Auxiliary Tasks.” *Proceedings of
the 38th International Conference on Machine Learning*, Proceedings of
machine learning research, vol. 139: 3329–39.
<https://proceedings.mlr.press/v139/fisch21a.html>.

</div>

<div id="ref-barber2023beyond" class="csl-entry">

Foygel Barber, Rina, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J.
Tibshirani. 2023. “Conformal Prediction Beyond Exchangeability.” *The
Annals of Statistics* 51 (2): 816–45.
<https://doi.org/10.1214/23-AOS2276>.

</div>

<div id="ref-gardner2023tableshift" class="csl-entry">

Gardner, Josh, Zoran Popovic, and Ludwig Schmidt. 2023. “Benchmarking
Distribution Shift in Tabular Data with TableShift.” *Advances in Neural
Information Processing Systems* 36: 53385–432.
<https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html>.

</div>

<div id="ref-gibbs2024online" class="csl-entry">

Gibbs, Isaac, and Emmanuel J. Candès. 2024. “Conformal Inference for
Online Prediction with Arbitrary Distribution Shifts.” *Journal of
Machine Learning Research* 25 (162): 1–36.
<https://www.jmlr.org/papers/v25/22-1218.html>.

</div>

<div id="ref-guo2023ehrshift" class="csl-entry">

Guo, Lin Lawrence, Ethan Steinberg, Scott Lanyon Fleming, et al. 2023.
“EHR Foundation Models Improve Robustness in the Presence of Temporal
Distribution Shift.” *Scientific Reports* 13 (1): 3767.
<https://doi.org/10.1038/s41598-023-30820-8>.

</div>

<div id="ref-kapash2025multiaccuracy" class="csl-entry">

Kapash, Daniel, Noam Barda, Omer Reingold, Noa Dagan, and Ran Balicer.
2025. “Multiaccuracy for Subpopulation Calibration over Distribution
Shift in Medical Prediction Models.” *Proceedings of the Sixth
Conference on Health, Inference, and Learning*, Proceedings of machine
learning research, vol. 287: 130–44.
<https://proceedings.mlr.press/v287/kapash25a.html>.

</div>

<div id="ref-levy2022autoupdating" class="csl-entry">

Levy, Todd J., Kevin Coppa, Jinxuan Cang, et al. 2022. “Development and
Validation of Self-Monitoring Auto-Updating Prognostic Models of
Survival for Hospitalized COVID-19 Patients.” *Nature Communications* 13
(1): 6812. <https://doi.org/10.1038/s41467-022-34646-2>.

</div>

<div id="ref-liu2023whyshift" class="csl-entry">

Liu, Jiashuo, Tianyu Wang, Peng Cui, and Hongseok Namkoong. 2023. “On
the Need for a Language Describing Distribution Shifts: Illustrations on
Tabular Datasets.” *Advances in Neural Information Processing Systems*
36: 51371–408.
<https://proceedings.neurips.cc/paper_files/paper/2023/hash/a134eaebd55b7406ff29cd75d5f1a622-Abstract-Datasets_and_Benchmarks.html>.

</div>

<div id="ref-liu2026longtail" class="csl-entry">

Liu, Shuqi, Jianguo Huang, and Luke Ong. 2026. “Conformal Prediction
Meets Long-Tail Classification.” *Proceedings of the AAAI Conference on
Artificial Intelligence* 40 (28): 23828–36.
<https://doi.org/10.1609/aaai.v40i28.39558>.

</div>

<div id="ref-nchsnhanes1999_2018" class="csl-entry">

National Center for Health Statistics. 2020a. *Continuous NHANES,
1999–2018: Questionnaires, Datasets, and Related Documentation*. Centers
for Disease Control; Prevention.
<https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/>.

</div>

<div id="ref-nchsnhanes2017demo" class="csl-entry">

National Center for Health Statistics. 2020b. *NHANES 2017–2018 Data
Documentation, Codebook, and Frequencies: Demographic Variables and
Sample Weights (DEMO_J)*. Centers for Disease Control; Prevention.
<https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/DEMO_J.htm>.

</div>

<div id="ref-nchsnhanes2017lead" class="csl-entry">

National Center for Health Statistics. 2020c. *NHANES 2017–2018 Data
Documentation, Codebook, and Frequencies: Lead, Cadmium, Total Mercury,
Selenium, and Manganese—Blood (PBCD_J)*. Centers for Disease Control;
Prevention.
<https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PBCD_J.htm>.

</div>

<div id="ref-niculescu2005goodprob" class="csl-entry">

Niculescu-Mizil, Alexandru, and Rich Caruana. 2005. “Predicting Good
Probabilities with Supervised Learning.” *Proceedings of the 22nd
International Conference on Machine Learning*, 625–32.
<https://doi.org/10.1145/1102351.1102430>.

</div>

<div id="ref-ojeda2023comparison" class="csl-entry">

Ojeda, Francisco M., Max L. Jansen, Alexandre Thiéry, et al. 2023.
“Calibrating Machine Learning Approaches for Probability Estimation: A
Comprehensive Comparison.” *Statistics in Medicine* 42 (29): 5451–78.
<https://doi.org/10.1002/sim.9921>.

</div>

<div id="ref-ovadia2019trust" class="csl-entry">

Ovadia, Yaniv, Emily Fertig, Jie Ren, et al. 2019. “Can You Trust Your
Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset
Shift.” *Advances in Neural Information Processing Systems* 32.
<https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html>.

</div>

<div id="ref-papaioannou2026robust" class="csl-entry">

Papaioannou, Jens-Michalis, Sebastian Jäger, Alexei Figueroa, et al.
2026. “Robust Conformal Prediction for Infrequent Classes.”
*Transactions on Machine Learning Research*.
<https://openreview.net/forum?id=nJ4p8rh3Ig>.

</div>

<div id="ref-park2020calibrated" class="csl-entry">

Park, Sangdon, Osbert Bastani, James Weimer, and Insup Lee. 2020.
“Calibrated Prediction with Covariate Shift via Unsupervised Domain
Adaptation.” *Proceedings of the Twenty Third International Conference
on Artificial Intelligence and Statistics*, Proceedings of machine
learning research, vol. 108: 3219–29.
<https://proceedings.mlr.press/v108/park20b.html>.

</div>

<div id="ref-podkopaev2021labelshift" class="csl-entry">

Podkopaev, Aleksandr, and Aaditya Ramdas. 2021. “Distribution-Free
Uncertainty Quantification for Classification Under Label Shift.”
*Proceedings of the Thirty-Seventh Conference on Uncertainty in
Artificial Intelligence*, Proceedings of machine learning research, vol.
161: 844–53. <https://proceedings.mlr.press/v161/podkopaev21a.html>.

</div>

<div id="ref-popordanoska2024lascal" class="csl-entry">

Popordanoska, Teodora, Gorjan Radevski, Tinne Tuytelaars, and Matthew B.
Blaschko. 2024. “LaSCal: Label-Shift Calibration Without Target Labels.”
*Advances in Neural Information Processing Systems* 37: 65386–414.
<https://doi.org/10.52202/079017-2088>.

</div>

<div id="ref-romano2020aps" class="csl-entry">

Romano, Yaniv, Matteo Sesia, and Emmanuel J. Candès. 2020.
“Classification with Valid and Adaptive Coverage.” *Advances in Neural
Information Processing Systems* 33: 3581–91.
<https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html>.

</div>

<div id="ref-roschewitz2025posthoc" class="csl-entry">

Roschewitz, Mélanie, Raghav Mehta, Fabio De Sousa Ribeiro, and Ben
Glocker. 2025. “Where Are We with Calibration Under Dataset Shift in
Image Classification?” *Transactions on Machine Learning Research*.
<https://openreview.net/forum?id=1NYKXlRU2H>.

</div>

<div id="ref-ruckart2021blrv" class="csl-entry">

Ruckart, Perri Zeitz, Robert L. Jones, Joseph G. Courtney, et al. 2021.
“Update of the Blood Lead Reference Value—United States, 2021.” *MMWR.
Morbidity and Mortality Weekly Report* 70 (43): 1509–12.
<https://doi.org/10.15585/mmwr.mm7043a4>.

</div>

<div id="ref-shi2024classwise" class="csl-entry">

Shi, Yuanjie, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao Doppa, and
Yan Yan. 2024. “Conformal Prediction for Class-Wise Coverage via
Augmented Label Rank Calibration.” *Advances in Neural Information
Processing Systems* 37: 132133–78.
<https://doi.org/10.52202/079017-4200>.

</div>

<div id="ref-steyerberg2004updating" class="csl-entry">

Steyerberg, Ewout W., Gerard J. J. M. Borsboom, Hans C. van Houwelingen,
Marinus J. C. Eijkemans, and J. Dik F. Habbema. 2004. “Validation and
Updating of Predictive Logistic Regression Models: A Study on Sample
Size and Shrinkage.” *Statistics in Medicine* 23 (16): 2567–86.
<https://doi.org/10.1002/sim.1844>.

</div>

<div id="ref-strack2014hba1c" class="csl-entry">

Strack, Beata, Jonathan P. DeShazo, Chris Gennings, et al. 2014. “Impact
of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
Clinical Database Patient Records.” *BioMed Research International*
2014: 781670. <https://doi.org/10.1155/2014/781670>.

</div>

<div id="ref-sun2023minimumrisk" class="csl-entry">

Sun, Zeyu, Dogyoon Song, and Alfred O. Hero. 2023. “Minimum-Risk
Recalibration of Classifiers.” *Advances in Neural Information
Processing Systems* 36: 69505–31.
<https://proceedings.neurips.cc/paper_files/paper/2023/hash/dbd6b295535e44f2b8ec0c3f1da7c509-Abstract-Conference.html>.

</div>

<div id="ref-tibshirani2019weighted" class="csl-entry">

Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel J. Candès, and Aaditya
Ramdas. 2019. “Conformal Prediction Under Covariate Shift.” *Advances in
Neural Information Processing Systems* 32.
<https://proceedings.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html>.

</div>

<div id="ref-uscensus2018acspums" class="csl-entry">

U.S. Census Bureau. 2018. *2018 ACS 1-Year Public Use Microdata Sample
(PUMS)*. U.S. Department of Commerce.
<https://www.census.gov/programs-surveys/acs/microdata/access/2018.html>.

</div>

<div id="ref-uscensus2018pumsdict" class="csl-entry">

U.S. Census Bureau. 2019. *2018 ACS 1-Year PUMS Data Dictionary*. U.S.
Department of Commerce.
<https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf>.

</div>

<div id="ref-usedcollegescorecarddocs2025" class="csl-entry">

U.S. Department of Education. 2025. *Technical Documentation: College
Scorecard Institution-Level Data*.
<https://collegescorecard.ed.gov/files/InstitutionDataDocumentation.pdf>.

</div>

<div id="ref-usedcollegescorecarddict2026" class="csl-entry">

U.S. Department of Education. 2026a. *College Scorecard Data
Dictionary*.
<https://collegescorecard.ed.gov/files/CollegeScorecardDataDictionary.xlsx>.

</div>

<div id="ref-usedcollegescorecard2026" class="csl-entry">

U.S. Department of Education. 2026b. *College Scorecard Data: June 10,
2026 Release*. <https://collegescorecard.ed.gov/data/>.

</div>

<div id="ref-vancalster2019achilles" class="csl-entry">

Van Calster, Ben, David J. McLernon, Maarten van Smeden, Laure Wynants,
Ewout W. Steyerberg, and Topic Group ‘Evaluating Diagnostic Tests and
Prediction Models’ of the STRATOS Initiative. 2019. “Calibration: The
Achilles Heel of Predictive Analytics.” *BMC Medicine* 17 (1): 230.
<https://doi.org/10.1186/s12916-019-1466-7>.

</div>

<div id="ref-vergouwe2017closed" class="csl-entry">

Vergouwe, Yvonne, Daan Nieboer, Rianne Oostenbrink, et al. 2017. “A
Closed Testing Procedure to Select an Appropriate Method for Updating
Prediction Models.” *Statistics in Medicine* 36 (28): 4529–39.
<https://doi.org/10.1002/sim.7179>.

</div>

</div>
