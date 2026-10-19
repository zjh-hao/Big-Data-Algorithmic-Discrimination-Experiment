# Research Extension V2

Read [中文结果](RESEARCH_EXTENSION_RESULTS.md), [English summary](RESEARCH_EXTENSION_RESULTS_EN.md), [baseline audit](BASELINE_AUDIT.md), [frozen protocol](PROTOCOL.md), [reproducibility](REPRODUCIBILITY_REPORT.md), and [limitations](LIMITATIONS.md).

Completed: six restored Home Credit fixed scoring models; 78 frozen policy configurations (63 feasible, 15 INFEASIBLE); 90 primary feasible-policy subgroup rows; 90 selective-label ledgers / 180 group estimates; 1000-repetition conditional paired intervals. Primary no-harm is infeasible for full, biased seed17, and biased seed61. For seeds29/43/89 it leaves the rule unchanged. No primary no-harm improvement is claimed.

The public baseline excludes checkpoints and ledgers. Historical 220 models / 701 rows / 9768 checks were checked as published records; the new 3707 record-level checks were actually run locally. Six narrowly restored models are new checkpoints, not recovered historical binaries. Historical tracked files and values remain unchanged. No remote push or PR.

## Run from repository root

Existing local run (no training):

```powershell
$env:PYTHONUTF8 = '1'
./.venv/Scripts/python.exe -m unittest discover -s research_extension_v2/tests -v
./.venv/Scripts/python.exe -m research_extension_v2.verify
./.venv/Scripts/python.exe -m research_extension_v2.report
```

Fresh clone/recovery:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m research_extension_v2.restore_data
./.venv/Scripts/python.exe -m unittest discover -s research_extension_v2/tests -v
./.venv/Scripts/python.exe -m research_extension_v2.run --output research_extension_v2/results/run_002
./.venv/Scripts/python.exe -m research_extension_v2.verify --run research_extension_v2/results/run_002
./.venv/Scripts/python.exe -m research_extension_v2.report --run research_extension_v2/results/run_002
```

`run` refuses an existing output directory. Re-evaluation does not need tree training: frozen scores, rules and records are already local under results/run_001. Full recovery trains only six fixed models if original legacy prediction files are absent. Do not run original training/report commands just to reproduce V2, since they write their own legacy outputs.

## Files

- `policy_noharm.py`: shared candidates, business/gap/groupwise protection, explicit infeasibility.
- `subgroup_audit.py`: training-only terciles, six intersections, Wilson intervals.
- `selective_label_ipw.py`: label-free exploration plans, observable-ledger estimator, separate offline oracle.
- `exploration_risk.py`: regular / exploration / actual risk.
- `statistics.py`: exact empirical joint-cell paired bootstrap, conditional DeLong AUC.
- `run.py`: isolated score recovery, freeze, evaluation, static replay.
- `verify.py`: independent grid optimization, metric/ledger reconstruction and legacy-hash checks.
- `calibration_diagnostic.py`: elementwise numerical checks for the local run.
- `results/run_001`: immutable scores, checkpoints, decisions, ledgers, identities and aggregate tables. Record-level data remain local.
- `FIGURES`: six PNG/PDF scientific figures.
- `presentation/output/Fairness_Research_V2.pptx`: 8-slide editable English deck.
- `presentation/SPEAKER_NOTES_EN.md`, `SPEAKER_NOTES_ZH.md`: English script, Chinese explanations, 480-second timing.
- `presentation/build_deck.mjs`, `content.json`, `evidence.json`: complete deck source and input evidence. Requires Codex bundled `@oai/artifact-tool` as documented in reproducibility.

## Table dictionary

All CSV metric rates are fractions unless named otherwise; multiply by100 for percent or percentage-point differences. `thin=1`, `group=1` denotes thin, `0` comparison. `epsilon` / `delta` are fractions. Main policies are business, gap_e0.02, noharm_e0.02_d0.01. AUC depends on the score, not the approval decision; an infeasible policy has no approval metrics, and its model AUC remains available in auc_intervals.csv.

`noharm_removed_candidates` counts the feasible-gap candidates eliminated by protection. `noharm_changes_gap_only=True` also includes a loss of feasibility; count only PASS rows when interpreting it as a different executable policy. `gap_changes_business` records whether adding the gap constraint changes the business rule. Feasibility and test violation flags are distinct.

`paired_intervals.csv`: `lo/hi` are conditional percentile intervals; `difference_lo/hi` are paired intervals relative to Business Group. The `difference` field stores the bootstrap mean difference, not the exact sample point contrast. Compute sample contrasts from COMPARISON_TABLES.csv, or subgroups.csv. `subgroup_0_1` means comparison / middle income. `subgroups.csv` contains point FRR, Wilson interval and baseline change; paired change intervals are keyed in paired_intervals.csv. Income boundaries:117000 /180000 in original income units, without claiming a currency conversion.

`ipw_estimates.csv`: `ess_good` is ESS among visible TARGET0 records, `ess_visible` among all visible records. The 0% exploration ESS does not repair nonidentifiability. Bootstrap errors compare estimator and oracle only in final offline evaluation. `exploration_risk.csv` stores conditional risk intervals and observed counts. Five seeds repeat one test population and must not be pooled as five real populations.

Statuses: PASS / FAIL are execution or verification states; INFEASIBLE is optimization; NOT_IDENTIFIABLE is identification; INSUFFICIENT_SUPPORT is estimator support. Optional Budgeted Exploration is NOT_RUN. Original unavailable feedback-ledger re-audit is BLOCKED_NOT_RUN. Do not substitute one status for another.
