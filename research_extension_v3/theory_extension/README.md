# Theory extension: business budgets, opportunity allocation and inverse pricing

The Chinese derivation is in `theory_note/main.tex` with empirical evidence in `theory_note/empirical.tex`. Open the `theory_note` directory in the built-in LaTeX viewer. All figures are generated from actual frozen experiment records.

## What was learned

- The exact minimum-harm feasible set has essentially no business allocation freedom: maximum validation subgroup opportunity range across the 9 primary scorers is only 0.0001544 percentage points.
- Explicit additional-harm budgets restore allocation freedom. However, normalized opportunity allocation makes a different tradeoff; it does not dominate the previous absolute-worst-group objective. Keep the existing primary approval method as the current empirical reference.
- The useful new contribution is the explicit two-budget formulation and **inverse opportunity pricing**: compute the least additional protection relaxation needed to raise the worst-group opportunity floor by a specified amount, while retaining the bank's risk and contribution constraints.
- The 27 pricing targets were fixed before their test replay. All are feasible on validation; test risk and contribution floors pass in 27/27, the 2pp group gap in 8/27. Worst-group opportunity improves relative to the frozen strict reference in 23/27, but only 9/27 reach the full requested increment.

## Sources and attribution

The building blocks are established: [DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types), [network shadow prices](https://web.stanford.edu/class/cs244/papers/ShadowPricesFairnessStability.pdf), [lexicographically fair learning](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2021.6), and [DKW–Massart](https://doi.org/10.1214/aop/1176990746). Nash bargaining already appears in [fairness-aware meta-learning](https://papers.nips.cc/paper_files/paper/2024/file/97a0f7d7fc2c26732943860241af1c7a-Paper-Conference.pdf). Our project-specific contribution is their explicitly defined lending formulation, mathematical specialization, implementation, business interpretation and falsifiable comparisons; no first-in-literature or SOTA claim is made.

## Modules

| File | Purpose |
|---|---|
| `allocation_policy.py` | Strict minimum-harm diagnostic; frozen run005 implementation. Its normalized numerical limitations are retained, not silently repaired. |
| `budget_allocation.py` | Explicit harm budget, same-set ablations, complete leximin with normalized stage checks. |
| `paired_certificate.py` | Validation-only simultaneous DKW audit; 8 groups × 9 frozen scorers. |
| `opportunity_price.py` | Validation-only inverse pricing for three fixed opportunity targets. |
| `evaluate_prices.py` | Replay all 27 frozen priced policies on test; no fitting or target selection. |
| `audit_leximin.py` | Separate business feasibility from normalized-stage numerical evidence. |
| `summarize.py` | Rebuild figures and empirical LaTeX appendix. |

## Finite runs

From the repository root, using `.venv-v3/Scripts/python.exe`:

```powershell
.venv-v3/Scripts/python.exe -m research_extension_v3.theory_extension.run
.venv-v3/Scripts/python.exe -m research_extension_v3.theory_extension.run_budget
.venv-v3/Scripts/python.exe -m research_extension_v3.theory_extension.opportunity_price --output research_extension_v3/results/run_006_budget/opportunity_prices
.venv-v3/Scripts/python.exe -m research_extension_v3.theory_extension.audit_leximin --run research_extension_v3/results/run_006_budget
.venv-v3/Scripts/python.exe -m research_extension_v3.theory_extension.summarize
.venv-v3/Scripts/python.exe -m unittest discover -s research_extension_v3/tests
```

All scorer training is reused from run001/run002. These experiments use validation for optimization and serialize strategies before test evaluation. The historical test has already been inspected in previous project development, so this is not a fresh blind holdout. The normalized objectives are not assumed to improve the absolute weakest group. The mathematical statements concern the exact empirical feasible set; numerical tolerances and future-population statistical assumptions are reported separately. Old run001–004 are retained unchanged.

## Results

- `../results/run_005_theory`: 99 rows; 9 scorers × 3 budgets × 3 objectives plus baselines. Independent numerical audit finds 3 normalized failures and 12 fully near-fixed leximin cases.
- `../results/run_006_budget`: 72 rows; 9 scorers × 2 harm budgets × 3 objectives plus baselines. 4,977 replay/metric checks pass; all 18 leximin stage audits pass.
- `../results/run_006_budget/opportunity_prices`: 27 finite validation pricing targets and complete frozen test replay; 2,160 independent checks pass.
- `../results/run_006_budget/figures`: three standalone figures for the report.

All 61 unit tests pass. There is no remaining model-training job in this extension.
