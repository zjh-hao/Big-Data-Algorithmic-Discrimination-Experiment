# 收益预算、群体保护与六子群最差误拒率：方法扩展实验

本报告只读取已冻结策略和测试重放结果。预测器保持原实验训练版本；新增内容是审批优化。本轮方法定位是课程项目中的业务、公平与群体保护联合设计，不宣称提出新的 SOTA 算法。

## 数据与比较口径

单位净贡献为正常标签获批数减去损失比乘以困难标签获批数；金额净贡献来自既定情景参数。TARGET=1 表示还款困难，指标不是实际银行账面利润。策略在 validation 上拟合后冻结；test 只重放。这个历史测试集曾被查看，不能称为新的盲测。

LP 策略对每个人给出获批概率，是单调分组阈值的随机混合。获批数、净贡献及误拒率表示期望值；风险是期望困难获批数除以期望总获批数，不是任一次随机执行都满足预算的保证。

原两组硬阈值与 LP 混合使用不同决策空间。LP 内的公平成本必须相对相同模型、种子、损失、风险与资金情景下的 LP 无公平约束收益基准计算；不能把决策空间和阈值网格变化带来的差异计作公平方法的单独贡献。

鲁棒变体在按申请 ID 固定分成的两个验证子集上同时施加相应约束。这是有限的验证稳定性检查，不是分布无关保证；下表保留测试超限情况。

主设置：风险预算 5%，公平容差 2 个百分点，原始严格保护目标 1 个百分点，收益预算 1%。自适应方法的有效保护上限由验证 LP 计算并另外报告，不能自动解释为满足原严格目标。

## 覆盖与可行性

登记需要的模型任务：24；表中已出现：24；缺失：无。

预设的严格六子群保护主方法，在 9 个完整数据模型/种子任务中只有 2 个验证可行。因此，‘同时满足全部严格要求’尚未普遍实现；不能只展示少数可行种子。这些冲突为后续计算最低必要保护松弛提供了直接依据。

| experiment | model | policy_id | robust | registered | feasible | infeasible |
| --- | --- | --- | --- | --- | --- | --- |
| main | lightgbm | adaptive_beta_0 | 0 | 3 | 1 | 2 |
| main | lightgbm | adaptive_beta_0.005 | 0 | 3 | 3 | 0 |
| main | lightgbm | adaptive_beta_0.02 | 0 | 3 | 3 | 0 |
| main | lightgbm | adaptive_beta_0.03 | 0 | 3 | 3 | 0 |
| main | lightgbm | adaptive_beta_0.05 | 0 | 3 | 3 | 0 |
| main | lightgbm | adaptive_primary | 0 | 3 | 3 | 0 |
| main | lightgbm | adaptive_robust | 1 | 3 | 0 | 3 |
| main | lightgbm | adaptive_robust_beta_0.02 | 1 | 3 | 3 | 0 |
| main | lightgbm | adaptive_robust_beta_0.03 | 1 | 3 | 3 | 0 |
| main | lightgbm | adaptive_robust_beta_0.05 | 1 | 3 | 3 | 0 |
| main | lightgbm | det_gap | 0 | 3 | 3 | 0 |
| main | lightgbm | det_gap_macro_protected | 0 | 3 | 1 | 2 |
| main | lightgbm | det_profit | 0 | 3 | 3 | 0 |
| main | lightgbm | lp_gap | 0 | 3 | 3 | 0 |
| main | lightgbm | lp_gap_robust | 1 | 3 | 3 | 0 |
| main | lightgbm | lp_profit_baseline | 0 | 3 | 3 | 0 |
| main | lightgbm | lp_profit_risk_robust | 1 | 3 | 3 | 0 |
| main | lightgbm | minimax_six | 0 | 3 | 1 | 2 |
| main | tabm | adaptive_beta_0 | 0 | 3 | 0 | 3 |
| main | tabm | adaptive_beta_0.005 | 0 | 3 | 3 | 0 |
| main | tabm | adaptive_beta_0.02 | 0 | 3 | 3 | 0 |
| main | tabm | adaptive_beta_0.03 | 0 | 3 | 3 | 0 |
| main | tabm | adaptive_beta_0.05 | 0 | 3 | 3 | 0 |
| main | tabm | adaptive_primary | 0 | 3 | 3 | 0 |
| main | tabm | adaptive_robust | 1 | 3 | 0 | 3 |
| main | tabm | adaptive_robust_beta_0.02 | 1 | 3 | 1 | 2 |
| main | tabm | adaptive_robust_beta_0.03 | 1 | 3 | 3 | 0 |
| main | tabm | adaptive_robust_beta_0.05 | 1 | 3 | 3 | 0 |
| main | tabm | det_gap | 0 | 3 | 3 | 0 |
| main | tabm | det_gap_macro_protected | 0 | 3 | 1 | 2 |
| main | tabm | det_profit | 0 | 3 | 3 | 0 |
| main | tabm | lp_gap | 0 | 3 | 3 | 0 |
| main | tabm | lp_gap_robust | 1 | 3 | 3 | 0 |
| main | tabm | lp_profit_baseline | 0 | 3 | 3 | 0 |
| main | tabm | lp_profit_risk_robust | 1 | 3 | 3 | 0 |
| main | tabm | minimax_six | 0 | 3 | 0 | 3 |
| main | xgboost | adaptive_beta_0 | 0 | 3 | 1 | 2 |
| main | xgboost | adaptive_beta_0.005 | 0 | 3 | 3 | 0 |
| main | xgboost | adaptive_beta_0.02 | 0 | 3 | 3 | 0 |
| main | xgboost | adaptive_beta_0.03 | 0 | 3 | 3 | 0 |
| main | xgboost | adaptive_beta_0.05 | 0 | 3 | 3 | 0 |
| main | xgboost | adaptive_primary | 0 | 3 | 3 | 0 |
| main | xgboost | adaptive_robust | 1 | 3 | 0 | 3 |
| main | xgboost | adaptive_robust_beta_0.02 | 1 | 3 | 3 | 0 |
| main | xgboost | adaptive_robust_beta_0.03 | 1 | 3 | 3 | 0 |
| main | xgboost | adaptive_robust_beta_0.05 | 1 | 3 | 3 | 0 |
| main | xgboost | det_gap | 0 | 3 | 3 | 0 |
| main | xgboost | det_gap_macro_protected | 0 | 3 | 2 | 1 |
| main | xgboost | det_profit | 0 | 3 | 3 | 0 |
| main | xgboost | lp_gap | 0 | 3 | 3 | 0 |
| main | xgboost | lp_gap_robust | 1 | 3 | 3 | 0 |
| main | xgboost | lp_profit_baseline | 0 | 3 | 3 | 0 |
| main | xgboost | lp_profit_risk_robust | 1 | 3 | 3 | 0 |
| main | xgboost | minimax_six | 0 | 3 | 1 | 2 |
| stress | lightgbm_biased | adaptive_primary | 0 | 5 | 5 | 0 |
| stress | lightgbm_biased | adaptive_robust | 1 | 5 | 0 | 5 |
| stress | lightgbm_biased | adaptive_robust_beta_0.03 | 1 | 5 | 5 | 0 |
| stress | lightgbm_biased | det_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_biased | det_gap_macro_protected | 0 | 5 | 4 | 1 |
| stress | lightgbm_biased | det_profit | 0 | 5 | 5 | 0 |
| stress | lightgbm_biased | lp_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_biased | lp_profit_baseline | 0 | 5 | 5 | 0 |
| stress | lightgbm_biased | minimax_six | 0 | 5 | 4 | 1 |
| stress | lightgbm_corrected | adaptive_primary | 0 | 5 | 5 | 0 |
| stress | lightgbm_corrected | adaptive_robust | 1 | 5 | 0 | 5 |
| stress | lightgbm_corrected | adaptive_robust_beta_0.03 | 1 | 5 | 5 | 0 |
| stress | lightgbm_corrected | det_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_corrected | det_gap_macro_protected | 0 | 5 | 4 | 1 |
| stress | lightgbm_corrected | det_profit | 0 | 5 | 5 | 0 |
| stress | lightgbm_corrected | lp_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_corrected | lp_profit_baseline | 0 | 5 | 5 | 0 |
| stress | lightgbm_corrected | minimax_six | 0 | 5 | 4 | 1 |
| stress | lightgbm_opportunity1 | adaptive_primary | 0 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | adaptive_robust | 1 | 5 | 0 | 5 |
| stress | lightgbm_opportunity1 | adaptive_robust_beta_0.03 | 1 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | det_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | det_gap_macro_protected | 0 | 5 | 2 | 3 |
| stress | lightgbm_opportunity1 | det_profit | 0 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | lp_gap | 0 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | lp_profit_baseline | 0 | 5 | 5 | 0 |
| stress | lightgbm_opportunity1 | minimax_six | 0 | 5 | 4 | 1 |

不可行记录全部保留；可行记录的均值不能替代完整任务的通过率。

## 主设置的同空间业务对照

| model | method_role | robust | utility | test_profit_retention | approval_rate | risk | abs_frr_gap | thin_frr | thick_frr | worst_subgroup_frr | utility_change_vs_lp | thin_frr_change_pp | thick_frr_change_pp | worst_subgroup_frr_change_pp | minimum_required_delta | effective_delta | feasible_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | adaptive_primary | 0.0000 | 35770.9755 | 0.9985 | 0.8224 | 0.0488 | 0.0122 | 0.1581 | 0.1459 | 0.1834 | -55.4642 | -3.0057 | 1.1645 | -3.3893 | 0.0110 | 0.0120 | 3.0000 |
| lightgbm | lp_gap | 0.0000 | 35809.4312 | 0.9995 | 0.8227 | 0.0487 | 0.0156 | 0.1602 | 0.1446 | 0.1857 | -17.0085 | -2.7970 | 1.0307 | -3.1605 | NA | NA | 3.0000 |
| lightgbm | lp_gap_robust | 1.0000 | 35211.1353 | 0.9828 | 0.7837 | 0.0449 | 0.0208 | 0.2013 | 0.1804 | 0.2319 | -615.3045 | 1.3099 | 4.6146 | 1.4661 | NA | NA | 3.0000 |
| lightgbm | lp_profit_baseline | 0.0000 | 35826.4397 | 1.0000 | 0.8232 | 0.0487 | 0.0539 | 0.1882 | 0.1343 | 0.2173 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 3.0000 |
| lightgbm | lp_profit_risk_robust | 1.0000 | 35264.2463 | 0.9843 | 0.7855 | 0.0450 | 0.0490 | 0.2204 | 0.1714 | 0.2537 | -562.1934 | 3.2212 | 3.7127 | 3.6411 | NA | NA | 3.0000 |
| lightgbm | minimax_six | 0.0000 | 35809.3752 | 0.9999 | 0.8209 | 0.0485 | 0.0075 | 0.1447 | 0.1522 | 0.1681 | -3.9314 | -0.6056 | 0.4596 | -0.5995 | NA | NA | 1.0000 |
| tabm | adaptive_primary | 0.0000 | 35243.1625 | 0.9991 | 0.8065 | 0.0483 | 0.0216 | 0.1811 | 0.1594 | 0.2041 | -31.9494 | -7.0430 | 2.6325 | -7.1229 | 0.0258 | 0.0268 | 3.0000 |
| tabm | lp_gap | 0.0000 | 35239.1906 | 0.9990 | 0.8068 | 0.0483 | 0.0247 | 0.1831 | 0.1584 | 0.2059 | -35.9214 | -6.8362 | 2.5303 | -6.9404 | NA | NA | 3.0000 |
| tabm | lp_gap_robust | 1.0000 | 34511.4698 | 0.9784 | 0.7697 | 0.0452 | 0.0225 | 0.2173 | 0.1947 | 0.2397 | -763.6421 | -3.4230 | 6.1644 | -3.5585 | NA | NA | 3.0000 |
| tabm | lp_profit_baseline | 0.0000 | 35275.1119 | 1.0000 | 0.8081 | 0.0484 | 0.1184 | 0.2515 | 0.1331 | 0.2753 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 3.0000 |
| tabm | lp_profit_risk_robust | 1.0000 | 34703.1510 | 0.9838 | 0.7756 | 0.0454 | 0.1051 | 0.2727 | 0.1676 | 0.2981 | -571.9609 | 2.1183 | 3.4531 | 2.2771 | NA | NA | 3.0000 |
| xgboost | adaptive_primary | 0.0000 | 35600.1608 | 0.9994 | 0.8175 | 0.0487 | 0.0185 | 0.1678 | 0.1492 | 0.1929 | -22.5385 | -1.5760 | 0.7878 | -1.7654 | 0.0071 | 0.0081 | 3.0000 |
| xgboost | lp_gap | 0.0000 | 35597.0294 | 0.9993 | 0.8177 | 0.0487 | 0.0213 | 0.1696 | 0.1483 | 0.1948 | -25.6699 | -1.3949 | 0.6958 | -1.5687 | NA | NA | 3.0000 |
| xgboost | lp_gap_robust | 1.0000 | 34996.7679 | 0.9824 | 0.7824 | 0.0454 | 0.0206 | 0.2029 | 0.1823 | 0.2331 | -625.9314 | 1.9396 | 4.0954 | 2.2539 | NA | NA | 3.0000 |
| xgboost | lp_profit_baseline | 0.0000 | 35622.6993 | 1.0000 | 0.8194 | 0.0489 | 0.0422 | 0.1835 | 0.1414 | 0.2105 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NA | NA | 3.0000 |
| xgboost | lp_profit_risk_robust | 1.0000 | 35126.0694 | 0.9861 | 0.7846 | 0.0454 | 0.0734 | 0.2397 | 0.1663 | 0.2761 | -496.6299 | 5.6174 | 2.4994 | 6.5583 | NA | NA | 3.0000 |
| xgboost | minimax_six | 0.0000 | 35613.8256 | 0.9977 | 0.8188 | 0.0488 | 0.0089 | 0.1461 | 0.1550 | 0.1720 | -83.2130 | -1.6126 | 0.9264 | -1.4388 | NA | NA | 1.0000 |

收益保持率使用测试重放的收益除以同模型 LP 基准的测试收益；收益预算约束是在验证集施加的。测试保持率高于 100% 或低于验证预算边界都作为观察结果呈现。只有正基准收益才计算比例。

## 决策空间与可行性

| model | method_role | decision_family | feasible | registered |
| --- | --- | --- | --- | --- |
| lightgbm | adaptive_primary | group_threshold_mixture | 3 | 3 |
| lightgbm | adaptive_robust | group_threshold_mixture | 0 | 3 |
| lightgbm | det_gap | deterministic_two_thresholds | 3 | 3 |
| lightgbm | det_gap_macro_protected | deterministic_two_thresholds | 1 | 3 |
| lightgbm | det_profit | deterministic_two_thresholds | 3 | 3 |
| lightgbm | lp_gap | group_threshold_mixture | 3 | 3 |
| lightgbm | lp_gap_robust | group_threshold_mixture | 3 | 3 |
| lightgbm | lp_profit_baseline | group_threshold_mixture | 3 | 3 |
| lightgbm | lp_profit_risk_robust | group_threshold_mixture | 3 | 3 |
| lightgbm | minimax_six | group_threshold_mixture | 1 | 3 |
| tabm | adaptive_primary | group_threshold_mixture | 3 | 3 |
| tabm | adaptive_robust | group_threshold_mixture | 0 | 3 |
| tabm | det_gap | deterministic_two_thresholds | 3 | 3 |
| tabm | det_gap_macro_protected | deterministic_two_thresholds | 1 | 3 |
| tabm | det_profit | deterministic_two_thresholds | 3 | 3 |
| tabm | lp_gap | group_threshold_mixture | 3 | 3 |
| tabm | lp_gap_robust | group_threshold_mixture | 3 | 3 |
| tabm | lp_profit_baseline | group_threshold_mixture | 3 | 3 |
| tabm | lp_profit_risk_robust | group_threshold_mixture | 3 | 3 |
| tabm | minimax_six | group_threshold_mixture | 0 | 3 |
| xgboost | adaptive_primary | group_threshold_mixture | 3 | 3 |
| xgboost | adaptive_robust | group_threshold_mixture | 0 | 3 |
| xgboost | det_gap | deterministic_two_thresholds | 3 | 3 |
| xgboost | det_gap_macro_protected | deterministic_two_thresholds | 2 | 3 |
| xgboost | det_profit | deterministic_two_thresholds | 3 | 3 |
| xgboost | lp_gap | group_threshold_mixture | 3 | 3 |
| xgboost | lp_gap_robust | group_threshold_mixture | 3 | 3 |
| xgboost | lp_profit_baseline | group_threshold_mixture | 3 | 3 |
| xgboost | lp_profit_risk_robust | group_threshold_mixture | 3 | 3 |
| xgboost | minimax_six | group_threshold_mixture | 1 | 3 |

可行性提升需要同时查看决策空间变化与约束差异；不能仅凭 LP 更易可行就声称算法全面优于硬阈值。

## 六个子群的机会变化

| model | method_role | robust | subgroup_0_0_frr_change_pp | subgroup_0_1_frr_change_pp | subgroup_0_2_frr_change_pp | subgroup_1_0_frr_change_pp | subgroup_1_1_frr_change_pp | subgroup_1_2_frr_change_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | adaptive_primary | 0.0000 | 1.1388 | 1.2289 | 1.1149 | -3.3893 | -3.0648 | -2.2725 |
| lightgbm | lp_gap | 0.0000 | 1.0004 | 1.1010 | 0.9786 | -3.1605 | -2.8313 | -2.1340 |
| lightgbm | lp_profit_risk_robust | 1.0000 | 3.6757 | 3.9484 | 3.4762 | 3.6411 | 3.2644 | 2.4503 |
| lightgbm | lp_gap_robust | 1.0000 | 4.6183 | 4.8817 | 4.3031 | 1.4661 | 1.3622 | 0.9701 |
| lightgbm | minimax_six | 0.0000 | 0.4670 | 0.5285 | 0.3732 | -0.6886 | -0.7023 | -0.3249 |
| tabm | adaptive_primary | 0.0000 | 2.6157 | 2.8758 | 2.3680 | -7.1229 | -7.7306 | -5.9068 |
| tabm | lp_gap | 0.0000 | 2.5243 | 2.7600 | 2.2710 | -6.9404 | -7.4854 | -5.7149 |
| tabm | lp_profit_risk_robust | 1.0000 | 3.4269 | 3.7759 | 3.1059 | 2.2771 | 2.2142 | 1.7110 |
| tabm | lp_gap_robust | 1.0000 | 6.0216 | 6.7093 | 5.6723 | -3.5585 | -3.6812 | -2.8186 |
| xgboost | adaptive_primary | 0.0000 | 0.7696 | 0.8811 | 0.6977 | -1.7654 | -1.6490 | -1.1501 |
| xgboost | lp_gap | 0.0000 | 0.6793 | 0.7808 | 0.6136 | -1.5687 | -1.4728 | -0.9881 |
| xgboost | minimax_six | 0.0000 | 0.9339 | 0.9964 | 0.8387 | -1.8014 | -1.7259 | -1.1292 |
| xgboost | lp_profit_risk_robust | 1.0000 | 2.5458 | 2.6407 | 2.2921 | 6.5583 | 5.6438 | 3.9920 |
| xgboost | lp_gap_robust | 1.0000 | 4.2507 | 4.3543 | 3.6481 | 2.2539 | 1.8765 | 1.5014 |

负值表示相对同空间收益基准少拒绝正常标签客户；正值表示增加。收入分组只用于目标与审计，审批概率仍来自信用记录组内单调的分数阈值混合。

## 验证约束在测试集的迁移

| model | method_role | robust | beta | constraint | evaluated | violations | violation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | lp_profit_baseline | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_profit_baseline | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | adaptive_primary | 0.0000 | 0.0100 | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | adaptive_primary | 0.0000 | 0.0100 | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | adaptive_primary | 0.0000 | 0.0100 | gap | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | adaptive_primary | 0.0000 | 0.0100 | protection | 3.0000 | 2.0000 | 0.6667 |
| lightgbm | adaptive_primary | 0.0000 | 0.0100 | profit floor | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap | 0.0000 | NA | gap | 3.0000 | 2.0000 | 0.6667 |
| lightgbm | lp_profit_risk_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_profit_risk_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | lp_gap_robust | 1.0000 | NA | gap | 3.0000 | 2.0000 | 0.6667 |
| lightgbm | det_profit | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | det_profit | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | det_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | det_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| lightgbm | minimax_six | 0.0000 | 0.0100 | risk | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | minimax_six | 0.0000 | 0.0100 | capital | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | minimax_six | 0.0000 | 0.0100 | gap | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | minimax_six | 0.0000 | 0.0100 | protection | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | minimax_six | 0.0000 | 0.0100 | profit floor | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | det_gap_macro_protected | 0.0000 | NA | risk | 1.0000 | 0.0000 | 0.0000 |
| lightgbm | det_gap_macro_protected | 0.0000 | NA | capital | 1.0000 | 0.0000 | 0.0000 |
| tabm | lp_profit_baseline | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_profit_baseline | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | adaptive_primary | 0.0000 | 0.0100 | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | adaptive_primary | 0.0000 | 0.0100 | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | adaptive_primary | 0.0000 | 0.0100 | gap | 3.0000 | 2.0000 | 0.6667 |
| tabm | adaptive_primary | 0.0000 | 0.0100 | protection | 3.0000 | 3.0000 | 1.0000 |
| tabm | adaptive_primary | 0.0000 | 0.0100 | profit floor | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap | 0.0000 | NA | gap | 3.0000 | 3.0000 | 1.0000 |
| tabm | lp_profit_risk_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_profit_risk_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | lp_gap_robust | 1.0000 | NA | gap | 3.0000 | 3.0000 | 1.0000 |
| tabm | det_profit | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | det_profit | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | det_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| tabm | det_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| tabm | det_gap_macro_protected | 0.0000 | NA | risk | 1.0000 | 0.0000 | 0.0000 |
| tabm | det_gap_macro_protected | 0.0000 | NA | capital | 1.0000 | 0.0000 | 0.0000 |
| xgboost | lp_profit_baseline | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_profit_baseline | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | adaptive_primary | 0.0000 | 0.0100 | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | adaptive_primary | 0.0000 | 0.0100 | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | adaptive_primary | 0.0000 | 0.0100 | gap | 3.0000 | 2.0000 | 0.6667 |
| xgboost | adaptive_primary | 0.0000 | 0.0100 | protection | 3.0000 | 3.0000 | 1.0000 |
| xgboost | adaptive_primary | 0.0000 | 0.0100 | profit floor | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap | 0.0000 | NA | gap | 3.0000 | 2.0000 | 0.6667 |
| xgboost | minimax_six | 0.0000 | 0.0100 | risk | 1.0000 | 0.0000 | 0.0000 |
| xgboost | minimax_six | 0.0000 | 0.0100 | capital | 1.0000 | 0.0000 | 0.0000 |
| xgboost | minimax_six | 0.0000 | 0.0100 | gap | 1.0000 | 0.0000 | 0.0000 |
| xgboost | minimax_six | 0.0000 | 0.0100 | protection | 1.0000 | 0.0000 | 0.0000 |
| xgboost | minimax_six | 0.0000 | 0.0100 | profit floor | 1.0000 | 0.0000 | 0.0000 |
| xgboost | lp_profit_risk_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_profit_risk_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap_robust | 1.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap_robust | 1.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | lp_gap_robust | 1.0000 | NA | gap | 3.0000 | 2.0000 | 0.6667 |
| xgboost | det_profit | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | det_profit | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | det_gap | 0.0000 | NA | risk | 3.0000 | 0.0000 | 0.0000 |
| xgboost | det_gap | 0.0000 | NA | capital | 3.0000 | 0.0000 | 0.0000 |
| xgboost | det_gap_macro_protected | 0.0000 | NA | risk | 2.0000 | 0.0000 | 0.0000 |
| xgboost | det_gap_macro_protected | 0.0000 | NA | capital | 2.0000 | 0.0000 | 0.0000 |

每项只统计可行策略中适用且有记录的检查。名义与双折约束的任务构成可能不同，整体比例只作描述；应结合逐模型、逐方法的分母和不可行数量判断。

## 面对不可行约束的最低必要损害方案

后续有限扩展采用两阶段设计：先在原风险、资金、公平差距和收益下限下，计算最低必要的最大子群误拒率恶化幅度 d*；然后将保护上限冻结为 d* 加预设 0.1 个百分点，在这个明确的保护范围内最小化六子群最差误拒率，再以收益作平局选择。所有数值来自验证集，测试结果不会改变保护上限。

在主设置的 9 个可行自适应任务中，有 7 个的最低必要保护容差已超过原先请求的 1 个百分点。因此该方案提供的是可量化、尽量小的保护松弛，不能描述为严格 1pp 保护已经实现。

| model | method_role | robust | minimum_required_delta | effective_delta | requested_delta | test_profit_retention | worst_subgroup_frr | worst_subgroup_frr_change_pp | thin_frr_change_pp | thick_frr_change_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | adaptive_primary | 0.0000 | 0.0110 | 0.0120 | 0.0100 | 0.9985 | 0.1834 | -3.3893 | -3.0057 | 1.1645 |
| tabm | adaptive_primary | 0.0000 | 0.0258 | 0.0268 | 0.0100 | 0.9991 | 0.2041 | -7.1229 | -7.0430 | 2.6325 |
| xgboost | adaptive_primary | 0.0000 | 0.0071 | 0.0081 | 0.0100 | 0.9994 | 0.1929 | -1.7654 | -1.5760 | 0.7878 |

先求最低损害再改善最差子群，是为了把不能同时满足的要求变成可解释的业务选择。业务需要共同查看 d*、收益保持率、六组机会变化和测试超限，不能只看总公平差距。

## 与标准公平方法和原硬阈值的实际比较

| model | method | utility | approval_rate | risk | abs_frr_gap | worst_subgroup_frr | thin_frr | thick_frr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | cost_sensitive_fairlearn | 35460.7985 | 0.7989 | 0.0464 | 0.0199 | 0.2091 | 0.1789 | 0.1687 |
| lightgbm | det_profit | 35841.3333 | 0.8218 | 0.0485 | 0.0614 | 0.2260 | 0.1950 | 0.1336 |
| lightgbm | det_gap | 35833.6667 | 0.8191 | 0.0481 | 0.0054 | 0.1813 | 0.1558 | 0.1505 |
| lightgbm | lp_profit_baseline | 35826.4397 | 0.8232 | 0.0487 | 0.0539 | 0.2173 | 0.1882 | 0.1343 |
| lightgbm | lp_gap | 35809.4312 | 0.8227 | 0.0487 | 0.0156 | 0.1857 | 0.1602 | 0.1446 |
| lightgbm | adaptive_primary | 35770.9755 | 0.8224 | 0.0488 | 0.0122 | 0.1834 | 0.1581 | 0.1459 |
| lightgbm | adaptive_robust_beta_0.03 | 35147.1454 | 0.7832 | 0.0450 | 0.0124 | 0.2240 | 0.1938 | 0.1839 |
| tabm | cost_sensitive_fairlearn | 34936.4285 | 0.7880 | 0.0465 | 0.0250 | 0.2241 | 0.2013 | 0.1763 |
| tabm | det_profit | 35254.6667 | 0.8070 | 0.0483 | 0.0934 | 0.2572 | 0.2340 | 0.1406 |
| tabm | det_gap | 35193.0000 | 0.8030 | 0.0479 | 0.0187 | 0.2050 | 0.1822 | 0.1635 |
| tabm | lp_profit_baseline | 35275.1119 | 0.8081 | 0.0484 | 0.1184 | 0.2753 | 0.2515 | 0.1331 |
| tabm | lp_gap | 35239.1906 | 0.8068 | 0.0483 | 0.0247 | 0.2059 | 0.1831 | 0.1584 |
| tabm | adaptive_primary | 35243.1625 | 0.8065 | 0.0483 | 0.0216 | 0.2041 | 0.1811 | 0.1594 |
| tabm | adaptive_robust_beta_0.03 | 34511.3354 | 0.7698 | 0.0452 | 0.0203 | 0.2381 | 0.2156 | 0.1952 |

Fairlearn 使用已保存的成本敏感、验证选择版本，容差固定为 2 个百分点。本比较展示已有冻结方法的真实测试结果；不同方法的优化目标、随机化方式与网格不同，不能据此直接宣称全面支配或论文 SOTA。

lightgbm：自适应主方法相对成本敏感 Fairlearn 的平均单位情景净贡献变化 +310.18，最差子群误拒率变化 -2.577 个百分点。相对原硬阈值公平规则，净贡献变化 -62.69，最差子群误拒率变化 +0.203 个百分点，对照组误拒率变化 -0.452 个百分点。

tabm：自适应主方法相对成本敏感 Fairlearn 的平均单位情景净贡献变化 +306.73，最差子群误拒率变化 -2.000 个百分点。相对原硬阈值公平规则，净贡献变化 +50.16，最差子群误拒率变化 -0.093 个百分点，对照组误拒率变化 -0.407 个百分点。

这些比较共同说明收益、最差子群与附带误拒损害之间存在权衡；方法的价值应按明确目标判断。

## 区间和随机执行的不确定性

`paired_intervals.csv` 中的区间是固定模型、固定审批规则下的申请人配对自助区间，反映样本重抽的不确定性；不能替代训练随机种子比较，也不能代表重新调参后的整体不确定性。`randomization_replays.csv` 另外反映逐申请人独立随机获批的一次实际执行波动。

与标准 Fairlearn 的配对区间保存在 `standard_method_paired_intervals.csv`。按预设方法、逐种子呈现，不按区间是否显著来筛选种子。

| job | policy | repetitions | utility_mean | utility_sd | risk_mean | risk_sd | risk_met_rate | gap_met_rate | profit_floor_met_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_s731 | adaptive_primary | 100.0000 | 35763.7400 | 46.2465 | 0.0487 | 0.0002 | 1.0000 | 0.9400 | 1.0000 |
| lightgbm_s731 | lp_gap | 100.0000 | 35854.1300 | 37.4435 | 0.0485 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |
| lightgbm_s732 | adaptive_primary | 100.0000 | 35761.0700 | 40.1148 | 0.0489 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| lightgbm_s732 | lp_gap | 100.0000 | 35772.5700 | 41.6976 | 0.0489 | 0.0001 | 1.0000 | 0.1500 | 1.0000 |
| lightgbm_s733 | adaptive_primary | 100.0000 | 35803.1100 | 57.4324 | 0.0487 | 0.0002 | 1.0000 | 1.0000 | 1.0000 |
| lightgbm_s733 | lp_gap | 100.0000 | 35813.7900 | 54.3448 | 0.0487 | 0.0002 | 1.0000 | 1.0000 | 1.0000 |
| lightgbm_s733 | minimax_six | 100.0000 | 35810.7400 | 21.4498 | 0.0485 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| tabm_s731 | adaptive_primary | 100.0000 | 35236.6500 | 48.1568 | 0.0482 | 0.0002 | 1.0000 | 0.0000 | 1.0000 |
| tabm_s731 | lp_gap | 100.0000 | 35276.7500 | 40.8598 | 0.0482 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |
| tabm_s732 | adaptive_primary | 100.0000 | 35203.9300 | 33.7939 | 0.0483 | 0.0001 | 1.0000 | 0.5500 | 1.0000 |
| tabm_s732 | lp_gap | 100.0000 | 35199.4600 | 23.3794 | 0.0483 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |
| tabm_s733 | adaptive_primary | 100.0000 | 35280.4600 | 33.3335 | 0.0482 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |
| tabm_s733 | lp_gap | 100.0000 | 35236.2800 | 38.4611 | 0.0484 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |
| xgboost_s731 | adaptive_primary | 100.0000 | 35699.8000 | 14.2623 | 0.0489 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| xgboost_s731 | lp_gap | 100.0000 | 35696.7000 | 17.4075 | 0.0489 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| xgboost_s731 | minimax_six | 100.0000 | 35614.3300 | 34.5537 | 0.0488 | 0.0001 | 1.0000 | 1.0000 | 1.0000 |
| xgboost_s732 | adaptive_primary | 100.0000 | 35455.4900 | 46.3355 | 0.0488 | 0.0002 | 1.0000 | 0.0300 | 1.0000 |
| xgboost_s732 | lp_gap | 100.0000 | 35449.6800 | 51.0175 | 0.0488 | 0.0002 | 1.0000 | 0.0000 | 1.0000 |
| xgboost_s733 | adaptive_primary | 100.0000 | 35653.2500 | 35.6048 | 0.0483 | 0.0001 | 1.0000 | 0.1700 | 1.0000 |
| xgboost_s733 | lp_gap | 100.0000 | 35642.1700 | 31.5835 | 0.0484 | 0.0001 | 1.0000 | 0.0000 | 1.0000 |

满足比例来自保存的有限随机执行次数，只描述这些执行；它不是银行未来每一批贷款满足约束的保证。

## 偏差压力测试覆盖

| model | policy_id | robust | registered | feasible |
| --- | --- | --- | --- | --- |
| lightgbm_biased | adaptive_primary | 0 | 5 | 5 |
| lightgbm_biased | adaptive_robust | 1 | 5 | 0 |
| lightgbm_biased | adaptive_robust_beta_0.03 | 1 | 5 | 5 |
| lightgbm_biased | det_gap | 0 | 5 | 5 |
| lightgbm_biased | det_gap_macro_protected | 0 | 5 | 4 |
| lightgbm_biased | det_profit | 0 | 5 | 5 |
| lightgbm_biased | lp_gap | 0 | 5 | 5 |
| lightgbm_biased | lp_profit_baseline | 0 | 5 | 5 |
| lightgbm_biased | minimax_six | 0 | 5 | 4 |
| lightgbm_corrected | adaptive_primary | 0 | 5 | 5 |
| lightgbm_corrected | adaptive_robust | 1 | 5 | 0 |
| lightgbm_corrected | adaptive_robust_beta_0.03 | 1 | 5 | 5 |
| lightgbm_corrected | det_gap | 0 | 5 | 5 |
| lightgbm_corrected | det_gap_macro_protected | 0 | 5 | 4 |
| lightgbm_corrected | det_profit | 0 | 5 | 5 |
| lightgbm_corrected | lp_gap | 0 | 5 | 5 |
| lightgbm_corrected | lp_profit_baseline | 0 | 5 | 5 |
| lightgbm_corrected | minimax_six | 0 | 5 | 4 |
| lightgbm_opportunity1 | adaptive_primary | 0 | 5 | 5 |
| lightgbm_opportunity1 | adaptive_robust | 1 | 5 | 0 |
| lightgbm_opportunity1 | adaptive_robust_beta_0.03 | 1 | 5 | 5 |
| lightgbm_opportunity1 | det_gap | 0 | 5 | 5 |
| lightgbm_opportunity1 | det_gap_macro_protected | 0 | 5 | 2 |
| lightgbm_opportunity1 | det_profit | 0 | 5 | 5 |
| lightgbm_opportunity1 | lp_gap | 0 | 5 | 5 |
| lightgbm_opportunity1 | lp_profit_baseline | 0 | 5 | 5 |
| lightgbm_opportunity1 | minimax_six | 0 | 5 | 4 |

压力任务使用既定偏置数据训练得到的保存分数；本扩展只重做冻结审批优化。各偏差条件应配对同条件的 LP 基准，不能与完整数据基准混算收益成本。

## 方法结论应怎样写

本轮回答的是：在固定风险、资金与可容忍收益损失下，能否降低最不利子群的正常客户误拒率，以及代价由谁承担。六子群 minimax、收益下限和相对基准保护共同构成可解释的审批设计。

阈值混合、线性规划和 minimax 都是已有思想；课程贡献是把它们与当前 Home Credit 的偏差机制、透明收益情景和业务预算结合，并做同空间消融与测试超限审计。若某些保护条件不可行、测试收益低于预算或双折约束未改善迁移，保留结果并说明适用条件。

## 输出文件

- `policy_metrics.csv`：所有登记策略，包括不可行任务。
- `protected_matched_lp_baselines.csv`：同空间收益和机会变化的配对明细。
- `protected_test_constraints.csv`：逐方法测试约束超限次数与分母。
- `figures/protected_*.png/.pdf`：可用于汇报的真实结果图。
- `figure_data/protected_*.csv`：每张图的可追溯源记录。
- `protected_figure_manifest.json`：输入、脚本、配置和输出校验值。

![Means across all available seeds of each fixed method/budget; incomplete or infeasible jobs remain in coverage tables.](figures/protected_01_budget_and_worst_group.png)

![Registered finite primary-budget rules, including infeasible cases. A larger feasible set is an optimization-space change.](figures/protected_02_policy_space_feasibility.png)

![Fixed primary budget; six groups are credit-history group crossed with training-derived income terciles.](figures/protected_03_six_group_changes.png)

![Descriptive checks of nominal and two-fold six-group minimax at the fixed primary budget, where available. Infeasible cases are tabulated separately.](figures/protected_04_test_constraint_transfer.png)

![Every saved feasible LP point, including dominated points; no best-on-test model or budget selection.](figures/protected_05_same_space_tradeoffs.png)

![Primary fixed 1% profit budget. Validation first minimizes the necessary worst-subgroup deterioration; a prespecified 0.1pp slack is then added before worst-FRR/profit optimization.](figures/protected_06_minimum_required_harm.png)

![Existing validation-frozen cost-sensitive Fairlearn at tolerance 0.02; LP and deterministic rules retain their own decision spaces. Large markers are seed means, small markers are all seeds.](figures/protected_07_standard_method_comparison.png)

