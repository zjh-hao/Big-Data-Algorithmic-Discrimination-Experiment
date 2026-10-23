# 与 V3 实验直接相关的资料

1. **Philadelphia Fed, One Threshold Doesn't Fit All: Tailoring Machine Learning Predictions of Consumer Default for Lower-Income Areas**, WP 22-39/R，2024年10月修订。
   - 官方页面：https://www.philadelphiafed.org/consumer-finance/consumer-credit/one-threshold-doesnt-fit-all
   - 全文：https://www.philadelphiafed.org/-/media/FRBP/Assets/working-papers/2022/wp22-39.pdf
   - 本项目采用其成本敏感净贡献思路：正常贷款带来收益，困难贷款承担相对更大的损失；比较模型升级与公平成本。该文在自身美国信用卡数据上校准 lambda，不能把其参数视为 Home Credit 已观测的财务事实。

2. **Kozodoi, Jacob & Lessmann, Fairness in Credit Scoring: Assessment, Implementation and Profit Implications**, European Journal of Operational Research, 297(3), 1083–1094。
   - 作者论文：https://arxiv.org/html/2103.01907
   - 实验代码：https://github.com/kozodoi/Fair_Credit_Scoring
   - 该研究包括 Home Credit，比较公平处理、贷款收益与公平前沿。本项目参考其业务评价组织方式，并采用自己的冻结划分与分组。两者分数不能直接当作同协议榜单排名。

3. **MAS-led industry consortium, Veritas Toolkit 2.0 / bank implementation cases**, 2023。
   - 官方入口及白皮书附件：https://www.mas.gov.sg/news/media-releases/2023/toolkit-for-responsible-use-of-ai-in-the-financial-sector
   - 用于说明银行业在实际 AI 评估中同时考虑公平、业务风险、透明度与治理。该实践不为本项目的利率、损失率提供逐笔校准值。

4. **Gorishniy et al., TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling**, ICLR 2025。
   - 作者代码：https://github.com/yandex-research/tabm
   - 论文：https://arxiv.org/abs/2410.24210
   - V3 直接加载作者模型源码，保留成员独立损失和概率平均；使用本项目特征、有限 reference 搜索和4060训练。

5. **Pereira, Valdrighi & Raimundo, M²FGB: A Min-Max Gradient Boosting Framework for Subgroup Fairness**, FAccT 2025。
   - 作者代码及引用：https://github.com/hiaac-finance/m2fgb
   - 论文版本代码：https://github.com/hiaac-finance/m2fgb/tree/facct25
   - V3 使用 facct25 核心原始定义，剥离无关方法的导入；标签映射、早停、预处理与参数搜索均在元数据中登记。

6. **Fairlearn, ThresholdOptimizer**。
   - 源码：https://github.com/fairlearn/fairlearn
   - 接口与公平约束说明：https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html
   - V3 分别使用原生准确率目标与成本加权准确率；后者通过困难标签行的整数复制实现，数学等价性有执行检查。

7. **LightGBM**。
   - 官方项目：https://github.com/lightgbm-org/LightGBM
   - V3 实际运行版本4.6.0，参数与 reference 搜索结果见每个模型任务的 JSON。

8. **Home Credit Default Risk**。
   - 原数据入口：https://www.kaggle.com/competitions/home-credit-default-risk/data
   - 本项目使用公开镜像恢复文件，原镜像提交与SHA256在 `data_source_recovery.json` 中记录。TARGET 的说明来自原数据列说明文件。

五个下载仓库的固定提交、历史归档SHA256和关键文件校验见 [upstream_sources.json](upstream_sources.json)，可通过 `setup_baselines.py` 获取并核验；实际环境与算法文件校验见每次运行的 `reproducibility.json`。`Fair_Credit_Scoring` 为业务评价参考，未称为完整复现。

9. **Fairlearn postprocessing 与 SciPy HiGHS LP（新增方法依据）**。
   - 随机阈值后处理说明：https://fairlearn.org/main/user_guide/mitigation/postprocessing.html
   - SciPy线性规划接口：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
   - 单调阈值随机混合是既有方法思想。run_003/run_004将其扩展为本项目的收益底线、六子群最小必要伤害与有限验证稳定性设计，不将随机化或线性规划本身宣称为新算法。实际运行SciPy1.14.1，结果和求解重试保存在冻结策略JSON中。

10. **跨领域资源分配与统计理论**。
    - Ghodsi等，DRF，NSDI 2011：https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types
    - Kelly等，网络影子价格与比例公平，1998：https://web.stanford.edu/class/cs244/papers/ShadowPricesFairnessStability.pdf
    - Diana等，Lexicographically Fair Learning，FORC 2021：https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2021.6
    - Zeng等，Fairness-Aware Meta-Learning via Nash Bargaining，NeurIPS 2024：https://papers.nips.cc/paper_files/paper/2024/file/97a0f7d7fc2c26732943860241af1c7a-Paper-Conference.pdf
    - Massart，DKW紧常数，1990：https://doi.org/10.1214/aop/1176990746
    - 理论归属、迁移范围、推导和反例见 [main.tex](theory_extension/theory_note/main.tex)。既有定理工具本身不称原创，双预算信贷构造与逆向定价为本项目的具体研究设计。
