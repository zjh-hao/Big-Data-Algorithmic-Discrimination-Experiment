# 英文汇报可直接使用的实验摘要

## Slide 1 — Where does the data come from?

- Real Home Credit data; controlled selective under-sampling.
- Thin-file borrowers: 0–1 observable bureau records.
- Reweighting reduced thin-file false rejections by 17.74 percentage points in the main scenario.
- This demonstrates a possible mechanism, not misconduct by the data provider.

Figure: `figures/01_mechanism.png`

## Slide 2 — What does the system optimize?

- The risk scores are fixed; approval objectives change.
- Compare accuracy, risk-constrained approvals, and approvals with a fairness constraint.
- For full-data scores, the 2-pp constraint is non-binding: it changes no decisions.
- For biased scores, the same constraint reduces the mean gap from 4.20 to 0.89 pp; approvals change from 81.80% to 81.66%.
- Validation constraints are not test guarantees; report both groups and risk.

Figure: `figures/06_objectives.png`

## Slide 3 — How does the system update?

- Five rounds of selective-label historical replay; five simulation seeds.
- All branches retain independent labeled validation support; each batch is assumed to mature before the next update.
- Approved-only updates vs. a frozen model: thin-file FRR changes by -16.44 pp.
- Exploration + IPW vs. approved-only updates: -2.57 pp.
- Post-primary diagnostic: approved-only vs. matched-size random feedback, +3.56 pp.
- Under these assumptions, no amplification was observed from the initial state; selective feedback still trails the matched-size random reference.

Figure: `figures/08_feedback_trajectory.png`

## Slide 4 — What does mitigation cost?

- Exploration reveals more outcomes by approving some otherwise rejected applicants.
- Actual risk including exploration: 5.28% in this replay.
- Weighting can improve representation but increases estimation variance.
- Local CPU implementation of an AWS-inspired workflow; no production or causal claims.

Figure: `figures/10_exploration_cost.png`
