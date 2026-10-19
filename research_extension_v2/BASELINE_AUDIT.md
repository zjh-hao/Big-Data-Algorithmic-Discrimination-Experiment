# 基线审查（2026-10-19）

## 身份与范围

指定仓库本机未检出，已克隆到当前工作目录下同名子目录。来源 HEAD `5a43e3758933f3440b6c53a1dd2224407c62577d`，初始 git status clean，分支 `feat/fairness-research-v2`。无 AGENTS.md，无 research_extension/ 或V1包。已完整阅读请求列出的报告、协议、四个核心代码文件，另读全部现有方法测试、数据恢复与配置。原文件不改写；基线哈希和新增结果分开保存。

## 当前实现与证据

| 层次 | 当前模块及运行 | 已发布证据 | 不可从发布物独立重算的部分 |
|---|---|---|---|
| 原始A | src/experiment.py，31个采样模型 | results/metrics.csv、data_audit.json、paired_intervals.csv | 原始数据、模型、预测、划分未发布 |
| 原始B | src/extensions.py::objective_module，6模型×7策略 | results/extensions/objective/metrics.csv，42行 | 逐条分数和冻结规则未发布 |
| 原始C | 同脚本五轮选择性回放与追加诊断 | feedback/test_metrics、stream_metrics、diagnostic_metrics | 台账、成熟标签、检查点未发布 |
| audit A/B/C | audit_extension/run_experiment.py，固定200树depth5 | metrics.csv实际701行、model去重220；A41、B30、C180、C_shadow450 | models、policies、training、ledgers、predictions、frozen_all、identity均未发布 |
| 历史核验 | audit_extension/verify.py | verification.json记录passed=true、9768检查、520唯一策略 | 这是发布者历史核验记录，本机克隆不能据此声称已重新核验9768项 |

运行方式为README中run_complete.ps1与audit_extension/run.ps1；完整重建会恢复原数据并重训所有模块。本轮仅恢复六个预设模型（或若发现分数则直接复用），后处理不重训。

## 结果与适用边界

当前audit结果：50%偏置下thin FRR40.27%，同组人数比例保持对照20.62%，纠正21.24%；机会权重1将差距28.79pp变为1.30pp，同时comparison FRR11.48%变为16.28%。full上的同权重将差距6.28pp变为7.40pp。历史原始B用400树，audit用200树，不跨轮差值归因。本轮恢复环境为macOS CPU，不默认等同历史Windows数值。

## 复用与必须恢复

P0：若有validation/test预测即可，无需任何树训练；现在发布物仅汇总表，不能由汇总数反推阈值，需恢复六个固定模型与分数。P1还需训练收入和测试收入按ID连接，不需新模型。P2/P3若有原反馈台账可直接审计；现无台账，改用独立静态标签可见性回放，原日志审计标BLOCKED_NOT_RUN，不能伪称重建了历史多轮路径。收入界和split只用原确定性函数重建到新目录。可选Budgeted Exploration为NOT_RUN。

## 历史缺口更新

AUDIT_REPORT里的纯人数/标签混杂由audit A的group_only解决；未改变训练损失由audit B加权交叉熵解决；始终外部验证由audit C的external/observed/frozen及同模型shadow对照解决。不能继续把这三项写成不存在。机会损失仍是代理目标，不保证FRR；无双侧no-harm、有限收入子群、FRR估计的IPW可信度比较，仍待本轮扩展。

## 泄漏与比较检查

代码中未发现测试标签用于拟合/阈值选参；原始划分以group×label分层是既定设计。词表仅原训练池拟合；反馈保留角色不进训练；未揭示标签在台账为空。历史同模型业务与fair策略采用同候选空间，也以gap破同分，故business并非完全不使用公平信息。原始objective模块将分数转float32会合并极近分数，本轮保持float64完整存取，并重建共同空间，不能与旧表伪作精确同一空间。

现有方法没有证明真实时点所有字段可得、自然人不跨集合、真实拒绝者反事实或未来公平。历史测试已看过，不是盲测。5%为验证经验预算；随机探索可能超预算。历史五seed为同人群的随机重复。参考分布纠正依赖有代表性标签。未校正多重探索比较。上述限制仍存在，未发现不等于证明无泄漏。
