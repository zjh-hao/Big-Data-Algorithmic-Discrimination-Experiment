# Speaker script and timing

Total: 8 minutes (480 seconds).

## 1. Algorithmic Discrimination in Credit Approval (0:00, 40 s)

Today I examine algorithmic discrimination through a credit approval experiment. The question is broader than whether a model predicts repayment difficulties accurately. We also need to know which applicants receive opportunities, how a fairness intervention redistributes those opportunities, and whether the available feedback lets us measure the result reliably. I use Home Credit historical loan records and a fixed XGBoost scoring framework. The contribution is a controlled and reproducible engineering study, inspired by established research. It does not establish unlawful discrimination, and it does not represent a production lending system. I will first explain the research background, then present feasible solutions and their measured costs.


## 2. Background and Fairness Challenges (0:40, 60 s)

Credit approval combines a prediction with a decision. In this experiment, TARGET zero denotes the historical normal label, while TARGET one denotes recorded repayment difficulties. The thin group has at most one observable bureau record. The comparison group has at least two. Our main opportunity measure is the false rejection rate: the share of normal-label records that the policy rejects. A small difference between two group averages does not tell us whether either group became worse off. Moreover, repayment feedback is selective because rejected applicants usually have no observed loan outcome. These issues span the entire cloud data lifecycle, from collection and processing to model training, approval rules, and feedback monitoring. Our implementation validates this lifecycle locally on a CPU.

References: R1, R5

## 3. Related Work and Research Gaps (1:40, 65 s)

Seven papers motivate the study. Hardt and colleagues provide the opportunity-based fairness perspective. Corbett-Davies and colleagues formulate fairness and utility as constrained decision problems. Agarwal and colleagues show how explicit fairness constraints connect to cost-sensitive classification. These ideas motivate a clear separation between business objectives and fairness requirements. Kearns and colleagues explain why fairness for a few broad groups can hide disparities inside intersections. Lakkaraju and colleagues identify the selective labels problem. Kilbertus and colleagues emphasize learning decision policies when predictions and observed outcomes are imperfect. Valdrighi and colleagues examine long-term fairness with selective labels. I borrow these questions and design principles. I do not claim to reproduce their reductions, adversarial subgroup learning, or reinforcement learning algorithms.

References: R1, R2, R3, R4, R5, R6, R7

## 4. Original Experimental Framework (2:45, 60 s)

The existing repository already studies biased sampling, training objectives, approval constraints, and feedback updates. I checked its current audit extension rather than treating an earlier audit as the latest state. Its published evidence contains two hundred twenty model identifiers and seven hundred one evaluation rows. However, the public repository omits the model checkpoints and record-level ledgers. I therefore restored only six prespecified scoring models from hash-verified data, using the original deterministic split. The training, reference, calibration, validation, and test proportions remain fifty, ten, ten, ten, and twenty percent. All policies for all six models freeze before test evaluation. The test records were examined previously, so this is a follow-up mechanism study, not a new blind test.


## 5. Fairness–Performance Trade-offs (3:45, 75 s)

The full-model comparison shows why the fairness objective needs careful interpretation. Business Group produces a thin-group false rejection rate of twenty point one five percent and a comparison-group rate of fourteen point three two percent. Adding only the gap constraint reduces the thin-group rate to sixteen point one four percent, but raises the comparison-group rate to sixteen point zero five percent. The absolute gap falls from five point eight three to zero point zero nine percentage points. Approval falls by about zero point two six percentage points, while test approved risk remains about four point seven nine percent. The model and its AUC are identical. These changes come entirely from decision rules. A smaller gap therefore describes a redistribution of opportunity, rather than evidence that everyone benefits.

References: R2

## 6. Proposed Fairness Improvements (5:00, 70 s)

My main proposed solution adds a groupwise no-harm constraint. We maximize approval while keeping validation approved risk below five percent, the absolute false rejection gap below two percentage points, and each group's deterioration below one percentage point relative to the same-model business rule. Every strategy uses the same threshold grid and deterministic tie handling. The result is informative even when the optimization fails. Under the primary parameters, three of the six models have no feasible grid solution. The other three select exactly the same rule as gap-only. We do not relax the constraints or disguise failure as rejecting everyone. Sensitivity analysis retains every prespecified epsilon and delta. The engineering value is exposing the feasibility boundary and the cost of protection, rather than guaranteeing an improvement.

References: R1, R2, R3

## 7. Experimental Evaluation and Feasibility (6:10, 70 s)

The remaining audits address who may lose and how reliably we can evaluate that loss. Six income intersections use boundaries estimated only from training data. In the full-model gap-only policy, the middle-income comparison subgroup's false rejection rate rises by about one point eight two percentage points. This is an exploratory finding. For selective labels, we freeze the original planned decisions, randomly approve some planned rejections, and keep planned approval distinct from actual approval. At ten percent exploration, thin-group naive false rejection is only about two point four nine percent, while IPW estimates twenty point three three percent against a twenty point one five percent offline oracle. But information costs risk: overall actual approved risk rises from about four point seven nine to five point one eight percent. Zero exploration leaves rejected-label fairness unidentified.

References: R4, R5, R6, R7

## 8. Conclusions and Limitations (7:20, 40 s)

The practical lesson is to evaluate fairness interventions as constrained decisions with explicit costs and reliable monitoring. A reduced group gap can coexist with harm to another group or an internal subgroup. No-harm constraints can be infeasible. Exploration can improve label coverage while increasing lending risk. Our independent checks reconstruct the policies and saved record-level arithmetic, but they do not establish future fairness. The study still lacks real rejected-applicant counterfactuals, temporal validation, and person-level clustering. The cloud architecture is a feasible extension proposal. We completed local CPU validation, with no AWS deployment or distributed performance benchmark. These boundaries are part of the result.

References: R6, R7

## References

- R1: Hardt et al., NeurIPS 2016. https://proceedings.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html
- R2: Corbett-Davies et al., KDD 2017. https://www.kdd.org/kdd2017/papers/view/algorithmic-decision-making-and-the-cost-of-fairness
- R3: Agarwal et al., ICML 2018. https://proceedings.mlr.press/v80/agarwal18a.html
- R4: Kearns et al., ICML 2018. https://proceedings.mlr.press/v80/kearns18a.html
- R5: Lakkaraju et al., KDD 2017. https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf
- R6: Kilbertus et al., AISTATS 2020. https://proceedings.mlr.press/v108/kilbertus20a.html
- R7: Valdrighi et al., ICML 2026. https://proceedings.mlr.press/v306/valdrighi26a.html

Script: 946 English words. Target pace: 118 words/minute. Timing is a rehearsal plan, not measured audio.
