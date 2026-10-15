# Presentation notes — audit extensions

## Data: representation vs. label composition

- Match total sample size; then match both group sizes.
- Group-only sampling preserves within-group outcome proportions.
- Thin-file FRR, group-only minus random: -0.13pp; conditional 95% CI [-0.77, +0.52].
- Thin-file FRR, distorted labels minus group-only: +19.64pp; conditional 95% CI [+18.75, +20.49].
- Controlled mechanism evidence; not proof of lender misconduct.

## Training: which errors receive weight?

- Train XGBoost with standard, cost-sensitive, opportunity-weighted and combined log loss.
- Keep records, tree settings, calibration and approval selection rules fixed.
- Opportunity weighting is a surrogate objective, not an equal-FRR constraint.
- Biased data, opportunity1 minus standard FRR: -25.29pp; conditional 95% CI [-26.28, -24.26].
- Report AUC, both groups, approval rates and risk together.

## Feedback: where do validation labels come from?

- Cross two acquisition policies with three validation information sources.
- Independent matched-size, observed-only, and frozen initial holdouts.
- Holdout roles assigned before observing labels; never train on holdout records.
- Selected-only, independent minus observed-only FRR: +4.07pp; conditional 95% CI [-1.89, +9.25].
- Same final observed-only model, changing only validation information: +3.79pp; conditional 95% CI [-0.79, +8.38].
- Outcomes are assumed to mature before each update; actual exploration risk is reported separately.

## Scope

- 220 additional CPU models; all parameters frozen before new evaluation.
- Previously viewed historical test set; conditional paired intervals, no external blind validation.
- Neither low FRR nor small group gaps alone establishes a sound credit policy.
