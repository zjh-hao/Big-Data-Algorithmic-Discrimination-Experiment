# 正式文献、思想与逐页对应

核对日期：2026-10-19。以下都是论文思想依据，不表示完整复现论文算法。项目贡献是固定模型上的工程组合及可复核审计。

| ID | 正式论文与主来源 | 提供的思想 | 本项目实现与差别 | PPT对应 |
|---|---|---|---|---|
| R1 | Hardt, Price, Srebro. Equality of Opportunity in Supervised Learning. NeurIPS 2016. https://proceedings.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html | 基于有利真实结果的机会准则，以及评分后的决策调整 | 本项目把TARGET0视为历史有利标签，审计FRR；有限确定性阈值网格并非论文完整最优随机后处理 | 2,3,6 |
| R2 | Corbett-Davies, Pierson, Feller, Goel, Huq. Algorithmic Decision Making and the Cost of Fairness. KDD 2017, 797–806. https://www.kdd.org/kdd2017/papers/view/algorithmic-decision-making-and-the-cost-of-fairness | 业务目标与公平限制联合约束优化，机会调整可能有成本 | 业务代理为批准人数与TARGET1比例预算，不是论文的公共安全目标或银行利润 | 3,5,6 |
| R3 | Agarwal, Beygelzimer, Dudik, Langford, Wallach. A Reductions Approach to Fair Classification. ICML 2018, PMLR80:60–69. https://proceedings.mlr.press/v80/agarwal18a.html | 公平分类可写成受约束问题，通过代价敏感子问题求解 | 借鉴显式约束与目标区分；本项目未实现reductions、成本敏感oracle或随机分类器混合 | 3,6 |
| R4 | Kearns, Neel, Roth, Wu. Preventing Fairness Gerrymandering: Auditing and Learning for Subgroup Fairness. ICML 2018, PMLR80:2564–2572. https://proceedings.mlr.press/v80/kearns18a.html | 少数粗群体的公平可以掩盖交叉子群差异 | 本项目只审计6个预设征信×收入交叉组；没有重现无限子群审计、弱学习oracle或对抗学习 | 3,6,7 |
| R5 | Lakkaraju, Kleinberg, Leskovec, Ludwig, Mullainathan. The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables. KDD 2017, 275–284. https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf | 决策影响结果是否可见，从而扭曲评价与比较 | 本项目使用受控历史标签隐藏，区别observed与oracle；真实拒贷反事实仍未识别 | 2,3,7 |
| R6 | Kilbertus, Gomez Rodriguez, Schölkopf, Muandet, Valera. Fair Decisions Despite Imperfect Predictions. AISTATS 2020, PMLR108:277–287. https://proceedings.mlr.press/v108/kilbertus20a.html | 选择性标签下应关注决策政策、公平约束和探索，而不只追求预测准确 | 本项目固定政策的探索与估计审计，没有重现其直接随机政策学习 | 3,7,8 |
| R7 | Valdrighi, Valera, Raimundo. Long-term Fairness with Selective Labels. ICML 2026, PMLR306:123279–123310. https://proceedings.mlr.press/v306/valdrighi26a.html | 长期决策与人群动态中，朴素可见标签公平可能偏离真实公平；论文结合观察与预测构建框架并提出RL方法 | 本项目未实现其长期公平框架或RL；静态IPW回放仅检查有限条件下的估计，不能验证长期公平 | 3,7,8 |

PPT中R编号与每页speaker notes对应；详细引用和URL同时保留在英文讲稿中。各论文摘要仅作上述范围的思想总结，细节结论不据此扩张为本项目证据。

## 数据与流程

- Home Credit Default Risk，原始入口：https://www.kaggle.com/competitions/home-credit-default-risk/data
- 本次固定镜像与SHA-256：../data/raw/source.json，恢复核对：results/data_recovery.json。
- AWS流程参考：https://github.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection 。本项目没有实际AWS部署。
- 历史实验：../audit_extension/RESULTS_REPORT.md、results/metrics.csv与verification.json（历史记录）。
