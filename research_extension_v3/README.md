# Home Credit：公平约束的业务成本与收益导向审批

本目录是 2026-10-22 根据老师反馈实现并运行的 V3 实验。核心问题：数据偏差怎样造成误拒；公平约束如何影响批准数量、贷款金额、风险和净贡献；强模型与收益导向审批能否降低公平成本。

**2026-10-23：本轮全部优化、复现和双预算理论扩展已完成。** 优先阅读 [统一研究更新与复现顺序](../RESEARCH_UPDATES_2026-10-22_23.md)，然后查看 [理论扩展](theory_extension/README.md)、[机会定价结果](results/run_006_budget/opportunity_prices/opportunity_prices.csv) 和 [测试重放](results/run_006_budget/opportunity_prices/test_evaluation.csv)。归一化分配未全面优于原绝对最差组主方案；没有 SOTA 声明。发布前全仓库97项测试通过。

原有限后台队列已全部完成，`results/run_002` 包含扩大搜索、三种子、强模型偏置压力测试及业务敏感性；自动跟进已停止。查看 [后续结果](results/run_002/FOLLOWUP_RESULTS_CN.md)。

2026-10-22 又按“自己的方法是否真正完成”补做两轮：`run_003` 审计严格子群保护的可行性；`run_004` 实现收益预算内的最小必要伤害审批。最新内容见 [方法与结果报告](results/run_004/METHODS_RESULTS_CN.md)、[优化完成情况](OPTIMIZATION_AUDIT_CN.txt)、[独立校验](results/run_004/protected_verification.json)。run_004 完成24个冻结分数任务、297条策略记录，37,629项独立检查通过；不可行和测试超限仍保留。预测器不重训，全部新增决策在 validation 上拟合后冻结。

首轮结果与图表仍在 [run_001](results/run_001/RESULTS_CN.txt)，没有覆盖。各运行的图表位于各自 `figures/`，输入点与文件指纹位于 `figure_data/` 和 manifest。

## 复现入口

在仓库根目录运行 PowerShell：

Git只保存历史汇总证据，逐申请人预测、权重和缓存留在本地。新克隆从头复现前，应在专用副本里将已发布 `results` 保留为 `results_published`，不能把缺少本地检查点的新克隆当作断点恢复；完整步骤见根目录研究更新。

```powershell
./run_research_v3.ps1 setup   # 新环境：Python 3.12、CUDA PyTorch 与 CPU 依赖
./run_research_v3.ps1 run     # 完整实验；已完成模型任务按检查点跳过
./run_research_v3.ps1 report  # 只重建统计与图表
./run_research_v3.ps1 verify  # 只独立校验

# 新增方法（已完成任务会跳过；不要覆盖原运行的冻结计划）
.venv-v3/Scripts/python.exe -m research_extension_v3.protected_run
.venv-v3/Scripts/python.exe -m research_extension_v3.adaptive_run
.venv-v3/Scripts/python.exe -m research_extension_v3.protected_compare
.venv-v3/Scripts/python.exe -m research_extension_v3.protected_verify --run research_extension_v3/results/run_004
.venv-v3/Scripts/python.exe -m research_extension_v3.protected_report --run research_extension_v3/results/run_004 --require-complete
```

首次取得数据可以沿用仓库的 `src/prepare_data.py`；其三个文件的固定 SHA256 在原项目中登记。V3 重新下载后的记录在 `data_source_recovery.json`。官方来源版本和关键文件校验见 [upstream_sources.json](upstream_sources.json)。setup会取得运行必需的三个固定提交，默认保存在仓库旁的 `credit-fairness-baselines`；可通过环境变量 `CREDIT_FAIRNESS_BASELINES` 修改。执行 `python -m research_extension_v3.setup_baselines --check-only` 可核验，`--all` 包含两个可选参考仓库。适配层保留上游核心算法。

## 数据与选择规则

使用完整申请表 307,511 条记录，以及申请前可观测的 bureau 聚合。主分组为征信记录数 ≤1 的薄信用记录人群和对照人群；该分组不是法律上的受保护属性认定。

固定原项目 `split_seed=20261013`，train/reference/calibration/validation/test 为 50/10/10/10/20%。训练集拟合特征编码、插补与归一化；reference 用于超参数选择和早停；calibration 用于 Platt 概率校准；validation 选择审批规则；test 只计算冻结规则的结果。历史留出集已在此前项目中查看，本轮不宣称新增盲测。

所有模型共享申请人 ID 与原始可用特征。树模型使用 262 列独热编码及原生缺失处理；LR/TabM 另用训练集拟合的中位数插补与缺失指示，合计 334 列。TabM 在训练集上对连续变量做正态分位数变换，二值变量保持不变。

## 模型与对照

- LR：C=0.1/1，两种配置按 reference AUC 选择。
- XGBoost：沿用 200 棵树、深度5、学习率0.05，作为原模型基准。
- LightGBM：3 个叶子数/叶节点样本配置，reference AUC 早停。
- TabM：直接加载 ICLR 2025 作者实现；3 个深度/宽度配置在首个种子上按 reference AUC 选择，余下种子复用所选配置；GPU 训练、独立成员 BCE、预测取成员概率均值。
- M²FGB：直接执行 FAccT 2025 `facct25` 分支原始类与函数；以正常还款为正类使用 TPR 公平代理损失，输出统一为困难概率。跳过原文件中无关模型的依赖导入，并限制 CPU 线程，核心函数未改写。
- M²FGB tuned：保留原 fair_weight=0.5 结果，另在 0.05/0.1/0.5 中按 reference AUC 选择后复用配置。
- Fairlearn：直接使用下载的官方源码与 `ThresholdOptimizer`，机会均等，正类为正常标签；保存期望审批概率。原生准确率目标与业务成本加权目标分别报告。

每个预测模型按 731/732/733 运行。LR 的 lbfgs 和当前无行列随机抽样的 M²FGB 为确定性配置，三个种子的输出可能完全相同；不把相同输出当作三份独立证据。

## 业务模型与政策空间

主情景净贡献：`U = 正常标签获批数 - lambda × 困难标签获批数`，lambda 为 3/5/7/10，主值5。

金额情景：`Pi = Σ a_i L_i [(1-y_i)r - y_i*loss - c]`。本金为 `AMT_CREDIT` 原单位；r=0.12、c=0.03，loss=`lambda*(r-c)-c`，使等额本金时与主情景一致。收益和损失属于情景参数，TARGET=1 是还款困难标签，并非逐笔已实现损失。

同一冻结概率网格下比较：统一收益阈值、分组最大批准数、分组最大净贡献、净贡献+公平差距限制、净贡献+公平+各组保护。主风险预算5%，补充3%/8%；公平误拒差距容差0/1/2/3/5/10个百分点；金额情景增加共同资金预算。用 validation 的经验净贡献选择阈值，另报告校准概率计算的期望净贡献。

同模型 `group_profit → group_profit_gap` 衡量公平成本；`common_profit → group_profit_gap` 还改变了阈值自由度，单列解释。跨模型对比同时改变预测能力，不归因于公平约束本身。

原生 Fairlearn 的准确率目标没有加入风险预算；业务加权 Fairlearn 保持标准算法，以困难样本整数复制精确实现成本敏感准确率，在 validation 上扫描成本并筛选风险预算，再计算 test 的期望业务指标。随机策略报告的是期望计数之比，单次随机实现可能不同。

## 偏见机制与训练目标

`mechanisms.py`：全数据基准，以及 5 个采样种子的总量匹配随机删样本、组人数完全匹配但保持标签比例、选择性减少 thin 正常样本、参考集重加权。匹配组人数的对照与偏置对照具有完全相同 thin/thick 人数，从而隔离标签构成失真。保存每条训练成员和权重，原始记录留在本地。

`training_objective.py`：直接复用原审计模块的机会损失权重，类别成本固定1，机会项强度0.5/1/4，在完整数据和5个偏置训练集上对照。包含负面结果，训练损失与审批目标分开解释。

## 不确定性与校验

`inference.py` 对固定模型、固定规则进行1,000次成对 bootstrap，利用 `(组别,标签,基准决定,候选决定,收入层)` 联合单元计数精确实现普通逐行重采样。输出总体差值区间与信用记录×收入6子群结果。区间不包含重新训练或重新选阈值的不确定性；金额收益只报告点估计。

`verify.py` 独立重建验证候选网格、检查有限网格最优性/不可行性、按逐人预测重算业务与公平指标、核对模型结果与 Fairlearn 期望决定、验证划分和输入指纹。不可行与测试预算超出明确保留。

主要文件：`prepare.py` 数据；`advanced_models.py` 官方算法适配；`advanced_search.py` reference 搜索；`business.py` 决策优化；`run.py` 运行；`fairlearn_business.py` 成本对齐基线；`mechanisms.py` 和 `training_objective.py` 偏见实验；`inference.py` 推断；`report.py` 图表；`verify.py` 独立校验。

## 收益预算内的最小必要伤害审批

`protected_policy.py` 将同一分数网格上的两组硬阈值扩展为单调阈值混合，用 SciPy HiGHS 求 LP。审批时仅用风险分数与信用记录组；收入三分位只用于六子群目标和保护约束。对每一信用组、每一候选阈值赋非负混合权重，权重和为1，因此组内风险越高获批概率不会增加。随机化思想已有文献与 Fairlearn 使用，本项目自己的设计是业务收益底线、最小必要伤害、六子群保护和有限稳定性检查的组合。

先在同一混合策略空间求收益最优基准，再要求净贡献至少保留其 `(1-beta)`。主要参数预先固定为 beta=1%、两组误拒差距不超过2pp、困难标签获批比例不超过5%。严格每个群体相对基准最多恶化1pp时，run_003只有2/9个主要模型种子可行。

自适应方法依次优化：①求满足收益、风险、公平要求所必需的最小最大子群恶化 d；②保护上限固定为 d+0.1pp，在此范围内最小化六子群最高误拒率；③保持前两阶段最优值，最大化净贡献。有效保护上限与原1pp目标分开记录，不能把放松后的可行性写成原目标全部达成。收益预算只扫描0/0.5/1/2/3/5%，不无限搜索。

`adaptive_run.py` 将预设方案应用于 LightGBM/TabM/XGBoost各三个种子以及三个训练偏差处理各五个种子。`protected_compare.py` 与既有成本敏感 Fairlearn 的固定2pp配置在同模型、同种子、同申请人上比较。`protected_verify.py` 不导入审批实现，独立重放序列化策略、重算业务指标并检查约束。35项单元测试包含联合阈值对 LP 独立最优值对照和真实求解数值故障回归。

均报告随机审批的期望结果；每个主要方法另作100次逐人随机执行与1,000次成对申请人 bootstrap，两种不确定性分开。稳健变体同时限制两个固定验证子集，1%收益预算全部不可行，3%时全部可行但业务规模和机会明显下降。验证约束不等于未见样本保证。当前优先用于汇报的是 LightGBM 主方案及其真实代价；更复杂的稳健方案作为对照。
