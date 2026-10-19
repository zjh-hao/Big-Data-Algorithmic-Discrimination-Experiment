# Research Extension V2：冻结协议

冻结日期：2026-10-19（Asia/Shanghai）。在新模型恢复及新测试评价前写入。历史测试已被查看，本轮是追加机制研究。

## 输入与恢复

基线提交 5a43e3758933f3440b6c53a1dd2224407c62577d；未发现 AGENTS.md 或 research_extension/V1。不修改任何原有跟踪文件。首先记录全部原跟踪文件的 SHA-256。数据按 src/prepare_data.py 固定 URL 和哈希恢复，但不调用其会覆盖 source.json 的 main。若本地有完整预测、划分、预处理器则优先读取并核对 ID，否则只恢复 full 和 50% 正常标签保留的五个 biased 模型（seed 17/29/43/61/89），而不重跑220个模型。模型固定沿用 audit_extension 的200树、depth5、seed731和4线程。原划分存在则核对复用，否则用原函数和 seed20261013 重建到新目录，不伪称恢复了原始检查点。全部报告按模型分别比较，不把新环境数值当作历史冻结数值。

## P0（主要分析）

评分、特征、预处理、校准集、验证集、测试集在同模型内完全相同。maximize批准人数，验证 TARGET=1 获批比例≤0.05，minimum=500。候选生成直接复用 threshold_points，每组 grid512，另含同模型统一业务阈值。候选只生成一次并哈希，三策略共享同一空间。包含完整分数 ties；相同分数只在同组内共同决定，组间不同阈值是显式策略。

A Business Group：只有风险约束；B Fair Gap Only：另有 abs(FRR_thin−FRR_comparison)≤epsilon；C Fair Gap + No-Harm：另有各组 FRR≤A同模型验证FRR+delta。主 epsilon=.02、delta=.01；所有epsilon [.01,.02,.05] × delta [0,.01,.02]均保留。不根据测试修改参数。所有策略共享批准人数优先、风险次之、差距再次、阈值索引最后的确定性破同分。业务策略用差距破同分，故估计的是公平硬约束的额外作用。

无解返回 INFEASIBLE 和 null thresholds；不生成全拒决策掩盖失败。记录风险/公平/保护过滤后的候选数、保护剔除数、是否改变gap-only解及各约束余量。只主张网格内最优。全部模型所有策略先冻结，然后统一读取测试评价。验证约束不等于测试或未来保证；另报测试违规。

## P1（探索性审计）

原完整训练池收入33⅓%、66⅔%经验分位数（numpy linear）固定边界，≤第一界low，≤第二界middle，其余high。六个征信×收入子群，收入不是法律保护属性。仅比较主参数三策略；同时输出两大组。n<100或TARGET0<100标 INSUFFICIENT_SUPPORT，空格缺失而非0。报告FRR Wilson95%区间、配对bootstrap变化区间、批准率/风险/人数。本审计不复现Kearns的无限子群对抗学习。

## P2（受控历史标签回放）

发布仓库没有逐条反馈日志，因此执行隔离的新静态可见性模拟，不冒充旧动态回放重建。固定每个P0模型的Business Group策略，测试人群只作封存后的离线回放。预先固定拒绝者探索率[0,.05,.10]、五模拟seed，offset850000；同seed各rate使用同一uniform，保证探索嵌套。决策函数没有标签参数，先冻结plan，再仅将actual_approval者的标签揭示给observable ledger。visible=actual_approval（假设结果均已成熟）；proposed保持原策略计划决策。inclusion_probability为proposed者1，拒绝者rate。

Naive仅用可见TARGET0计算原策略拒绝比例。IPW为可见TARGET0的1/pi加权拒绝数/加权TARGET0数。Oracle仅在独立最终离线函数中按ID访问历史标签，不进入估计函数、训练、校准或策略选择。任何目标组存在pi=0则IPW为NOT_IDENTIFIABLE；无TARGET0或ESS<100标INSUFFICIENT_SUPPORT。报告TARGET0 ESS（FRR分母相关）、所有可见记录ESS、最大权重、覆盖、误差和区间。所需条件是正确倾向、正性、随机探索/成熟标签机制正确、稳定回放标签和足够有效样本。比率估计有限样本可能有偏，不保证每个seed优于Naive。五seed不是五个真实场景。

## P3

复用P2可见台账分开统计常规、探索和实际总批准人数及风险。不按测试标签选rate，三个rate全部展示。Budgeted Exploration本轮NOT_RUN：不以不充分校准的预测风险假装未来预算保证。完整闭环预算控制仍未解决。

## 条件统计与复核

固定1000次配对非参数bootstrap，seed20261019。用记录的组别×标签×收入×完整策略批准向量联合单元进行multinomial重采样，等价于同一批记录配对bootstrap；不重新选策略或训练。P0输出各策略及相对A差值区间，P1输出配对变化；AUC用DeLong渐近95%区间，固定同模型策略AUC相同。P2用组别×oracle标签×可见性×proposed联合单元进行最终配对误差bootstrap，估计本身只用可见标签。P3同样重采样观察台账记录。报告五seed每个结果，不挑seed或rate，不以未校正探索性区间宣布一般显著。

区间条件于数据划分、模型、验证选择和模拟实现，不覆盖重新训练、自然人聚类、时间漂移、真实拒贷反事实或自适应阈值选择的不确定性。百分率表为%，绝对变化为pp。验证程序独立重建网格最优性、逐记录测试指标、子群、观察概率、IPW、风险和历史文件哈希。

## 状态

PASS实现/核验通过；FAIL发现不一致；INFEASIBLE无满足冻结约束的规则；NOT_RUN未执行的可选模块；BLOCKED_NOT_RUN必需输入不可用。NOT_IDENTIFIABLE是统计识别状态；INSUFFICIENT_SUPPORT是支持不足，两者不能写成运行失败或通过。
