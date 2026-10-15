# Home Credit 信贷风险、样本偏差与纠正实验（CPU）

## GitHub 仓库内容与从零复现

仓库包含实验代码、固定协议、中文报告、英文汇报摘要、PNG/PDF图表和汇总结果。原始数据、逐条申请预测/台账、训练ID、模型权重、虚拟环境和本机运行检查点不随仓库上传；它们可在本地运行生成。数据来源与获取方式见下文，第三方数据的获取与使用仍须遵守其原始条款。

在克隆后的仓库根目录，安装 Python 3.12 后运行：

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe -X utf8 src/prepare_data.py
./run_complete.ps1
./audit_extension/run.ps1
```

`run_complete.ps1` 生成基础实验所需的划分、词表和模型；随后执行补充实验。首次克隆后不能仅用报告模式核验逐记录结果，因为这些本地文件尚未生成。仓库已保存的汇总表和核验记录是本次已完成实验的证据；重新运行会生成当前机器的完整记录。目录内的旧本机绝对路径仅为历史操作示例，请使用克隆后的实际目录。


## 审查问题 1、2、4：新增完整对照

新增实验独立保存在 [audit_extension](audit_extension/README.md)，涵盖采样因素分离、验证来源反馈对照、训练损失对照。阅读新版 [补充实验结果报告](audit_extension/RESULTS_REPORT.md)，并结合旧版决策层实验理解三问。旧审查报告针对旧实验，其余限制继续成立。


审查更新（2026-10-14）：请先阅读 [原则与逻辑审查](AUDIT_REPORT.md)。当前结果是受控机制实验；反馈模拟依赖独立验证支持和逐批标签成熟假设，不构成真实歧视或生产有效性的证明。

本项目参考 AWS 官方信用风险预测与偏差分析流程，在本机运行 XGBoost，使用 Home Credit 真实贷款申请数据，研究“可观察信用记录较少的正常还款客户在训练中代表不足”造成的错拒，以及代表性参考数据是否有助于纠正。

**完整阅读入口：[三模块结果报告](COMPLETE_RESULTS_REPORT.md)。** 原 `RESULTS_REPORT.md` 保留模块 A 的完整结果。所有结果由代码生成，不预设纠正一定有效。

## 三个问题对应三个实验模块

| 模块 | 问题 | 实验 |
|---|---|---|
| A | 数据怎么来？ | 选择性采样、同规模随机对照、参考分布加权纠正 |
| B | 系统优化什么？ | 固定风险评分，对比准确率、业务风险预算、公平差距约束，共 42 个审批策略 |
| C | 系统如何更新？ | 五轮选择性标签回放，静态/仅获批反馈/随机探索/探索加权四路径，五个种子，共 80 个模型 |

主轨迹查看后另加 25 个同规模随机反馈诊断模型，以区分训练样本增长和选择机制。该诊断明确为后续分析，见 `FEEDBACK_DIAGNOSTIC.md`，不冒充事前登记实验。完整报告同时保留主四路径及诊断参照。

新增协议见 [EXTENSION_PROTOCOL.md](EXTENSION_PROTOCOL.md)，运行前固定。模块 B 是决策目标实验，没有改写 XGBoost 训练损失。模块 C 是历史回放模拟，没有真实线上放贷；随机探索的额外贷款及风险单独计入。原测试集已在 A 中查看，B/C 是后续机制分析而非全新外部盲测。

完整复现运行 `./run_complete.ps1`；仅重新生成汇报与核验运行 `./run_complete.ps1 -ReportOnly`。保留模块 A 的 `./run.ps1` 独立入口。报告首页总结各模块实际方向，不保证反馈一定放大偏差或所有修复都有效。

新增输出：

- `results/extensions/objective/`：42 个策略、冻结阈值、逐客户批准结果。
- `results/extensions/feedback/`：五轮观测台账（未观察标签为空）、训练 ID 和权重、测试轨迹。
- `models/extensions/`：80 个主模拟模型、25 个追加诊断模型、校准和阈值、每个种子的初始编码器。
- `figures/06_*.png` 至 `11_*.png`：目标权衡、反馈轨迹、代表性和探索成本，配套 PDF。
- `COMPLETE_RESULTS_REPORT.md`：中文完整报告；`PRESENTATION_NOTES.md`：可直接用于英文 PPT 的结果摘要。

## 与 AWS 官方案例的关系

[AWS 官方仓库](https://github.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection) 提供数据处理、XGBoost 训练、推理、解释、偏差审计与缓解流程。其原始数据是 South German Credit，缓解示例采用兼顾性能和公平性的优化。

本项目复用的是流程思路，代码独立编写：

| AWS 流程 | 本地对应实现 |
|---|---|
| Processing 数据处理 | 申请表与分块征信聚合、贷款记录级划分 |
| XGBoost 训练和调参 | CPU hist，4 线程，4 个预设候选 |
| 推理服务 | 本地批量预测文件，无在线部署 |
| Clarify / SHAP 解释与偏差审计 | 独立指标、XGBoost 原生 Tree SHAP |
| 偏差缓解与对照 | 选择性样本删减＋独立参考集联合分布加权 |

没有调用 AWS 服务、没有云资源费用；不能将实验结果描述为银行生产系统或 AWS 原实验结果。

## 数据来源

原始发布：[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)。

本次使用用户要求查找的 **GitHub 第三方镜像**：`qwyt/HomeCreditDefaultRisk` 的固定提交 `920bbfca12ac236a10ca6b9772286fa15f7a9cca`，其中 `dataset/full/app_train.parquet` 和 `bureau.parquet` 共约 58 MB。仓库的 `src_draft/load_dataset.py` 显示原始 CSV 直接转换为 Parquet。字段字典来自 `tejase13/Home-Credit-Default-Risk` 的固定提交。

程序校验 307,511 行 / 122 列的申请表、1,716,428 条唯一征信记录、原始标签数量、字段说明和关联关系。固定文件 SHA-256 见 `data/raw/source.json`。这些检查并不证明与 Kaggle 官方原始文件逐字节相同，也不把镜像仓库的许可当作原始数据的再授权。数据只作为本地课程研究输入，不随报告公开重分发。

如已从 Kaggle 获取原始 `application_train.csv` 和 `bureau.csv`，程序会优先读取 CSV。但不能在现有结果目录混用新输入，应复制代码到新的实验目录重新运行。

## 快速运行（PowerShell）

```powershell
cd 'E:\云计算汇报\loan_fairness_experiment_v2'
./run.ps1
```

当前目录已有独立 `.venv`。迁移到其他电脑时使用 Python 3.12：

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe src/prepare_data.py
./run.ps1
```

只检查文件可用性：`./run.ps1 -CheckOnly`。主程序仍会在正式运行时做完整结构和标签检查。

`prepare_data.py` 仅下载数据和文本参考，不运行第三方代码。网络中断可重新执行；已有输入会核对哈希。

## 固定实验协议

1. 分组：申请前可观察征信记录 0—1 条为薄弱组，至少 2 条为比较组；缺记录不等于现实中从未借款。
2. 贷款记录划分：训练 / 参考 / 校准 / 验证 / 测试 = 50 / 10 / 10 / 10 / 20%，组别×标签分层。
3. 特征：申请信息、征信聚合、收入负担比；排除 ID、性别、TARGET。数值缺失保留，类别词表仅拟合训练数据。
4. 完整训练基线选取 XGBoost 参数后冻结。随后 50%、25% 两档薄弱组 TARGET=0 留存率，各 5 个种子；每档每种子有同规模随机对照、偏置模型、纠正模型，共 31 个正式模型。
5. 纠正权重 `w(g,y)=P_reference(g,y)/P_biased(g,y)`；以参考集估计，不使用采样率直接给理想权重，也不强求群体具有相同风险。
6. 全部模型独立进行全局概率校准，在验证集按 5%（主）、3% / 8%（补充）风险预算确定统一阈值。至少 500 个验证获批者，没有可行阈值则不批准。验证风险约束不是测试或未来风险保证。
7. 先冻结全部模型、校准和阈值，再评估测试集。报告 AUC、Brier、校准偏差、获批率、获批风险、两组错拒率、有符号及绝对差距。
8. 配对 bootstrap 同步抽样测试贷款记录并抽样 5 个训练采样种子；区间条件于固定数据划分，不涵盖完整的人群/时间漂移不确定性。

标签 TARGET=1 的严格含义是“前 Y 期中至少一期逾期超过 X 天”，X、Y 未公布。不能将其直接转换为终身违约、损失金额或利润。

## 输出目录

```text
config.json                 冻结的种子、采样率、预算和模型候选
src/                        数据准备、实验、报告、独立核验
tests/                      采样、权重、划分、阈值及标签方向测试
data/raw/                   镜像数据、字段字典、来源与哈希
data/processed/             划分清单、每个模型的训练 ID 与权重
models/                     编码器、31 个 XGBoost 模型和决策参数
results/                    原始指标、差值区间、审计、运行与核验元数据
results/predictions/        每个模型的验证/测试逐客户预测
figures/                    5 组英文 PNG + PDF，适用于英文 PPT
references/                 AWS 说明及镜像转换代码的文本备份
RESULTS_REPORT.md           中文结果报告，包含图表与局限
```

程序顺序运行，限制 4 线程，使用 CPU，无 GPU 依赖。数据与代码/配置不变时会复用已完成模型；修改代码、配置或数据后会拒绝混用旧结果，请另建实验目录。模型文件仅加载本项目自己训练生成的版本。

逐客户预测概率原始计算精度为 float32。若从 CSV 重算边界审批判断，应先恢复 float32（`verify.py` 已处理），避免十进制文本转 float64 时把阈值上的样本误分到另一侧。

## 应如何表述结论

可以说：“在真实业务数据上，受控构造一种训练样本代表不足机制，检验它对风险评分及机会分配的影响，并评估针对该机制的修复。”

不能说：“证明 Home Credit 真实存在这种歧视”“加权使所有贷款公平”“复现了某银行的内部审批系统”。原始数据里被拒客户的还款反事实不可观察；有代表性的参考样本是方法能够应用的重要前提。
