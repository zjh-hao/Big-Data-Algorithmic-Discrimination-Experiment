# Research Extension V2

## Confirmed Experimental Findings

We recovered six prespecified CPU XGBoost scoring models and evaluated 78 frozen policies on the existing Home Credit historical test split. All 3707 independent checks passed. The full-model comparison gives test approval rates of 81.29% and 81.03% for Business Group and Fair Gap Only, with absolute FRR gaps of 5.83 and 0.09 percentage points. The full-model primary Fair Gap + No-Harm policy is INFEASIBLE. Primary no-harm is also infeasible for two of the five biased models. For the other three it selects the same rule as gap-only. No primary model shows an additional feasible no-harm improvement. The main constraints are a 5% validation approved-risk budget, a 2pp FRR-gap cap, and a 1pp maximum groupwise FRR increase relative to the same-model Business Group baseline. These are empirical validation constraints, not future guarantees. Across all presets, 15 policies were infeasible and 2 feasible no-harm policies changed the gap-only optimum in sensitivity configurations.

## Exploratory Findings

Seven sensitivity-policy test gaps exceed their frozen validation epsilon, despite validation feasibility. No test-label tuning follows these violations. Six training-tercile income intersections reveal how group averages can conceal local changes. Their paired intervals are conditional and unadjusted for multiple comparisons. Static selective-label replay evaluates 0%, 5%, and 10% exploration over five random seeds. At zero exploration, rejected-label outcomes have zero inclusion probability and group FRR is NOT_IDENTIFIABLE by IPW. With positive propensities, we compare observable-label naive and ratio-IPW estimates against a separate offline oracle, reporting groupwise ESS and estimation error. Actual exploration-inclusive risk exceeds 5% in 42 of 90 replay configurations. This count describes repeated simulation on the same records.

## Unresolved Questions

Can a calibrated exploration policy jointly manage information acquisition and total loan risk? Does groupwise validation protection survive temporal drift and finer subgroup definitions? We did not run Budgeted Exploration or a new longitudinal learning experiment. The unavailable original feedback ledgers prevent an independent record-level rerun of the historical 9768 checks.

## Limitations

The test set was previously inspected. Published historical aggregates remain unchanged, but these six restored models are new local checkpoints. Loan IDs do not establish person-level independence. TARGET is a historical repayment-difficulty indicator, not lifetime default or monetary loss. Historical borrowers do not identify real rejected-applicant counterfactuals. Five simulation seeds are not five real lending environments. Group and income definitions are audit strata, not a legal determination of discrimination. This is local CPU validation with no AWS deployment or distributed benchmark. Our contribution is a reproducible engineering combination inspired by established fairness literature, not a new theorem or a full reproduction of reductions, subgroup learning, or long-term reinforcement learning.
