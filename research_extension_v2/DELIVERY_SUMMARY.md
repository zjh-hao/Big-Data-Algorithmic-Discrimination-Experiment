# 最终交付汇总（A–H）

## A. 修改与新增

旧跟踪文件未修改，旧模型/报告/汇总均未覆盖。仅新增research_extension_v2：P0双侧保护策略、P1训练收入分组子群审计、P2可见台账IPW、P3探索风险、条件配对统计、独立核验、17项数学单测、6模型恢复编排、协议、基线审查、中英文报告、全部CSV、6组PNG/PDF及英文8页PPT源文件。完整逐文件身份在DELIVERY_MANIFEST.json。新数据按原URL及SHA恢复到data/raw，旧source.json不改。

## B. 实际测试

- `python -m compileall -q research_extension_v2`：PASS。
- `ruff check research_extension_v2 --select F821,F822,F823`：PASS。
- `python -m unittest discover -s tests -v`：10 PASS。
- `python -m unittest discover -s audit_extension -p test_methods.py -v`：3 PASS。
- `python -m unittest discover -s research_extension_v2/tests -v`：17 PASS。
- `python -m research_extension_v2.verify`：3707项独立检查PASS。
- `python -m research_extension_v2.calibration_diagnostic`：6校准器元素级梯度与有限区间检查PASS。
- 原跟踪文件`git diff --exit-code`：PASS。
- PPT：8页、8讲稿、2原生表格、3原生图表及嵌入数据工作簿通过结构/版式/字体核验，并从最终PPTX重导出8页逐页查看。未在PowerPoint应用实开。

日志位于results/，冻结运行身份、输入/输出SHA、具体环境和命令见REPRODUCIBILITY_REPORT.md。

## C. 已运行的真实实验

原始数据307511申请与1716428征信记录完整恢复。6个固定200树模型在本地CPU恢复，沿用原始确定性50/10/10/10/20划分。本轮恢复、冻结、测试及回放132.75秒，不包含下载/报告/核验。P0共78行，63可行、15 INFEASIBLE；主配置6模型中3无解、3与gap-only相同。P1有90个可评价子群行。P2/P3共90份静态回放台账、180组估计，其中60组零探索NOT_IDENTIFIABLE。全部预设配置保留。

历史发布物220模型、701行、9768核验记录已核对，但本机没有历史逐条检查点，不能宣称重跑那9768项。本轮与旧400树模型不直接相减。

## D. 主要科学发现

完整数据固定模型：gap-only把测试FRR差距从5.832pp降至0.090pp，thin从20.148%降至16.143%，comparison从14.316%升至16.053%。批准率81.289%变为81.029%，风险约4.79%。主no-harm不可行。

主no-harm整体3/6不可行，剩余3规则不变，因此没有主配置的额外可行改善。所有敏感性保留，另有两个epsilon5pp/delta1pp的可行配置改变规则；不据测试挑选。7行敏感性测试gap超出验证容差，全部报告。

探索性子群：完整模型comparison/middle FRR增加1.815pp，条件未校正95%区间[1.601,2.006]pp。有限6子群不是完整子群公平保证。

完整模型10%探索：thin naive2.488%、IPW20.328%、oracle20.148%（五seed均值），TARGET0 ESS约5139，最大权重10。0%探索因正性失败不可识别，即使可见样本ESS大。

探索成本：完整模型常规风险4.794%，5%探索实际风险4.989%，10%探索5.181%；10%探索批准子集风险约22.038%。均为五seed均值。全90回放42超5%，是同人群重复模拟，不是90真实业务场景。

## E. 新增价值

同评分同候选空间的No-Harm相对业务基线约束；显式区分约束过滤与实际改变选中规则；有限交叉子群损害检查；独立分离proposed/actual/visible/pi的FRR估计；并把探索实际放贷风险计入。是基于七篇正式论文思想的工程组合，没有声称新理论或复现reductions/对抗学习/RL。

## F. 未验证与不可识别

历史逐条反馈台账审计BLOCKED_NOT_RUN；可选Budgeted Exploration NOT_RUN。真实拒贷者还款反事实、长期人群反馈、自然人聚类、时间外泛化与未来风险保证未解决。已看过的历史测试不是盲测。当前没有AWS部署或分布式性能测量。全部边界见LIMITATIONS.md。

## G. 如何复现

已有本机输出优先`verify`和`report`，不训练。新机器安装requirements.txt、restore_data、运行单测，再`run --output research_extension_v2/results/run_002`，随后`verify/report --run`指向同目录。Windows完整PowerShell命令在README.md，macOS OpenMP说明及环境身份在REPRODUCIBILITY_REPORT.md。现有结果目录不可覆盖。

## H. 主要交付路径（相对本目录）

- 中文报告：RESEARCH_EXTENSION_RESULTS.md
- English summary：RESEARCH_EXTENSION_RESULTS_EN.md
- 完整P0表：COMPARISON_TABLES.csv
- P1/P2/P3、配对区间、AUC区间：results/run_001/*.csv
- 图表：FIGURES/（6组PNG/PDF）
- 英文PPT：presentation/output/Fairness_Research_V2.pptx
- 英文讲稿：presentation/SPEAKER_NOTES_EN.md
- 中文逐页说明：presentation/SPEAKER_NOTES_ZH.md
- PPT完整源：presentation/build_deck.mjs、content.json、evidence.json
- 文献逐页对应：REFERENCES.md
- 复现报告：REPRODUCIBILITY_REPORT.md

仓库绝对根：`/Users/pytenter/Documents/临时项目1/Big-Data-Algorithmic-Discrimination-Experiment`。
