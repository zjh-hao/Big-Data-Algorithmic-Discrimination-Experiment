# 中文逐页解释与时间分配

Total: 8 minutes (480 seconds).

## 1. Algorithmic Discrimination in Credit Approval (0:00, 40 s)

开场提出研究问题：公平干预何时有用、帮助谁、代价是什么、能否可靠评价。说明本项目是历史贷款数据上的受控工程研究，不是对现实机构违法歧视的认定。


## 2. Background and Fairness Challenges (0:40, 60 s)

解释TARGET方向、thin/comparison分组与FRR定义。公平覆盖数据采集、处理、训练、决策和反馈全过程。小差距不代表双方都好，可见标签也不代表完整人群。

References: R1, R5

## 3. Related Work and Research Gaps (1:40, 65 s)

七篇文献各有用途：机会准则、业务公平联合优化、约束分类、交叉子群与选择性标签、决策学习和长期公平。明确借鉴思想而非完整复现论文算法。

References: R1, R2, R3, R4, R5, R6, R7

## 4. Original Experimental Framework (2:45, 60 s)

尊重当前audit_extension已解决的缺口。历史公开701行、220模型，但缺逐条材料；本轮仅恢复6模型并复用分数，保持原50/10/10/10/20划分。历史测试已看过。


## 5. Fairness–Performance Trade-offs (3:45, 75 s)

展示完整数据模型的实际结果：gap-only让thin改善，但比较组FRR上升约1.74pp。差距下降不意味着所有人获益，批准率下降约0.26pp。三者评分相同，归因于后处理规则。

References: R2

## 6. Proposed Fairness Improvements (5:00, 70 s)

核心方法增加双侧保护，基线是同模型同候选空间business。主参数6模型中3个不可行，另3个规则不变。不可行为有价值的负面结论，不能放宽阈值或全拒掩盖。

References: R1, R2, R3

## 7. Experimental Evaluation and Feasibility (6:10, 70 s)

子群发现与估计可靠性同时展示：比较组中收入中等的子群FRR增加约1.82pp；10%探索下IPW接近oracle，naive严重低估。但探索后实际风险约5.18%，0探索不可识别。

References: R4, R5, R6, R7

## 8. Conclusions and Limitations (7:20, 40 s)

结论讲清约束、信息与风险共同治理。独立核验不等于现实有效性证明。缺少真实拒贷反事实、时间外验证和自然人聚类。云架构只是扩展提案，当前只有本地CPU验证。

References: R6, R7
