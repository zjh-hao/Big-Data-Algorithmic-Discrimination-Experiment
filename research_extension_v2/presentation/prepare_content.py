import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
e=json.loads((HERE/'evidence.json').read_text())
refs=[
('R1','Hardt et al., NeurIPS 2016','https://proceedings.neurips.cc/paper_files/paper/2016/hash/6a9659feb1216f14f7384ba499518b38-Abstract.html'),
('R2','Corbett-Davies et al., KDD 2017','https://www.kdd.org/kdd2017/papers/view/algorithmic-decision-making-and-the-cost-of-fairness'),
('R3','Agarwal et al., ICML 2018','https://proceedings.mlr.press/v80/agarwal18a.html'),
('R4','Kearns et al., ICML 2018','https://proceedings.mlr.press/v80/kearns18a.html'),
('R5','Lakkaraju et al., KDD 2017','https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf'),
('R6','Kilbertus et al., AISTATS 2020','https://proceedings.mlr.press/v108/kilbertus20a.html'),
('R7','Valdrighi et al., ICML 2026','https://proceedings.mlr.press/v306/valdrighi26a.html')]
speeches=[
'''Today I examine algorithmic discrimination through a credit approval experiment. The question is broader than whether a model predicts repayment difficulties accurately. We also need to know which applicants receive opportunities, how a fairness intervention redistributes those opportunities, and whether the available feedback lets us measure the result reliably. I use Home Credit historical loan records and a fixed XGBoost scoring framework. The contribution is a controlled and reproducible engineering study, inspired by established research. It does not establish unlawful discrimination, and it does not represent a production lending system. I will first explain the research background, then present feasible solutions and their measured costs.''',
'''Credit approval combines a prediction with a decision. In this experiment, TARGET zero denotes the historical normal label, while TARGET one denotes recorded repayment difficulties. The thin group has at most one observable bureau record. The comparison group has at least two. Our main opportunity measure is the false rejection rate: the share of normal-label records that the policy rejects. A small difference between two group averages does not tell us whether either group became worse off. Moreover, repayment feedback is selective because rejected applicants usually have no observed loan outcome. These issues span the entire cloud data lifecycle, from collection and processing to model training, approval rules, and feedback monitoring. Our implementation validates this lifecycle locally on a CPU.''',
'''Seven papers motivate the study. Hardt and colleagues provide the opportunity-based fairness perspective. Corbett-Davies and colleagues formulate fairness and utility as constrained decision problems. Agarwal and colleagues show how explicit fairness constraints connect to cost-sensitive classification. These ideas motivate a clear separation between business objectives and fairness requirements. Kearns and colleagues explain why fairness for a few broad groups can hide disparities inside intersections. Lakkaraju and colleagues identify the selective labels problem. Kilbertus and colleagues emphasize learning decision policies when predictions and observed outcomes are imperfect. Valdrighi and colleagues examine long-term fairness with selective labels. I borrow these questions and design principles. I do not claim to reproduce their reductions, adversarial subgroup learning, or reinforcement learning algorithms.''',
'''The existing repository already studies biased sampling, training objectives, approval constraints, and feedback updates. I checked its current audit extension rather than treating an earlier audit as the latest state. Its published evidence contains two hundred twenty model identifiers and seven hundred one evaluation rows. However, the public repository omits the model checkpoints and record-level ledgers. I therefore restored only six prespecified scoring models from hash-verified data, using the original deterministic split. The training, reference, calibration, validation, and test proportions remain fifty, ten, ten, ten, and twenty percent. All policies for all six models freeze before test evaluation. The test records were examined previously, so this is a follow-up mechanism study, not a new blind test.''',
'''The full-model comparison shows why the fairness objective needs careful interpretation. Business Group produces a thin-group false rejection rate of twenty point one five percent and a comparison-group rate of fourteen point three two percent. Adding only the gap constraint reduces the thin-group rate to sixteen point one four percent, but raises the comparison-group rate to sixteen point zero five percent. The absolute gap falls from five point eight three to zero point zero nine percentage points. Approval falls by about zero point two six percentage points, while test approved risk remains about four point seven nine percent. The model and its AUC are identical. These changes come entirely from decision rules. A smaller gap therefore describes a redistribution of opportunity, rather than evidence that everyone benefits.''',
'''My main proposed solution adds a groupwise no-harm constraint. We maximize approval while keeping validation approved risk below five percent, the absolute false rejection gap below two percentage points, and each group's deterioration below one percentage point relative to the same-model business rule. Every strategy uses the same threshold grid and deterministic tie handling. The result is informative even when the optimization fails. Under the primary parameters, three of the six models have no feasible grid solution. The other three select exactly the same rule as gap-only. We do not relax the constraints or disguise failure as rejecting everyone. Sensitivity analysis retains every prespecified epsilon and delta. The engineering value is exposing the feasibility boundary and the cost of protection, rather than guaranteeing an improvement.''',
'''The remaining audits address who may lose and how reliably we can evaluate that loss. Six income intersections use boundaries estimated only from training data. In the full-model gap-only policy, the middle-income comparison subgroup's false rejection rate rises by about one point eight two percentage points. This is an exploratory finding. For selective labels, we freeze the original planned decisions, randomly approve some planned rejections, and keep planned approval distinct from actual approval. At ten percent exploration, thin-group naive false rejection is only about two point four nine percent, while IPW estimates twenty point three three percent against a twenty point one five percent offline oracle. But information costs risk: overall actual approved risk rises from about four point seven nine to five point one eight percent. Zero exploration leaves rejected-label fairness unidentified.''',
'''The practical lesson is to evaluate fairness interventions as constrained decisions with explicit costs and reliable monitoring. A reduced group gap can coexist with harm to another group or an internal subgroup. No-harm constraints can be infeasible. Exploration can improve label coverage while increasing lending risk. Our independent checks reconstruct the policies and saved record-level arithmetic, but they do not establish future fairness. The study still lacks real rejected-applicant counterfactuals, temporal validation, and person-level clustering. The cloud architecture is a feasible extension proposal. We completed local CPU validation, with no AWS deployment or distributed performance benchmark. These boundaries are part of the result.'''
]
titles=['Algorithmic Discrimination in Credit Approval','Background and Fairness Challenges','Related Work and Research Gaps','Original Experimental Framework','Fairness–Performance Trade-offs','Proposed Fairness Improvements','Experimental Evaluation and Feasibility','Conclusions and Limitations']
times=[40,60,65,60,75,70,70,40]
zh=[
'开场提出研究问题：公平干预何时有用、帮助谁、代价是什么、能否可靠评价。说明本项目是历史贷款数据上的受控工程研究，不是对现实机构违法歧视的认定。',
'解释TARGET方向、thin/comparison分组与FRR定义。公平覆盖数据采集、处理、训练、决策和反馈全过程。小差距不代表双方都好，可见标签也不代表完整人群。',
'七篇文献各有用途：机会准则、业务公平联合优化、约束分类、交叉子群与选择性标签、决策学习和长期公平。明确借鉴思想而非完整复现论文算法。',
'尊重当前audit_extension已解决的缺口。历史公开701行、220模型，但缺逐条材料；本轮仅恢复6模型并复用分数，保持原50/10/10/10/20划分。历史测试已看过。',
'展示完整数据模型的实际结果：gap-only让thin改善，但比较组FRR上升约1.74pp。差距下降不意味着所有人获益，批准率下降约0.26pp。三者评分相同，归因于后处理规则。',
'核心方法增加双侧保护，基线是同模型同候选空间business。主参数6模型中3个不可行，另3个规则不变。不可行为有价值的负面结论，不能放宽阈值或全拒掩盖。',
'子群发现与估计可靠性同时展示：比较组中收入中等的子群FRR增加约1.82pp；10%探索下IPW接近oracle，naive严重低估。但探索后实际风险约5.18%，0探索不可识别。',
'结论讲清约束、信息与风险共同治理。独立核验不等于现实有效性证明。缺少真实拒贷反事实、时间外验证和自然人聚类。云架构只是扩展提案，当前只有本地CPU验证。'
]
slide_refs=[[],['R1','R5'],[r[0] for r in refs],[],['R2'],['R1','R2','R3'],['R4','R5','R6','R7'],['R6','R7']]
content=[];start=0
for j in range(8):
    refs_text='\n'.join(f'{a}: {b}. {c}' for a,b,c in refs if a in slide_refs[j])
    sources=f'Local evidence: research_extension_v2/results/run_001; historical audit_extension/results.\n{refs_text}'
    content.append(dict(title=titles[j],seconds=times[j],start=start,speech=speeches[j],zh=zh[j],references=slide_refs[j],sources=sources));start+=times[j]
(HERE/'content.json').write_text(json.dumps(content,indent=2,ensure_ascii=False),encoding='utf-8')
for language in ['EN','ZH']:
    text='# '+('Speaker script and timing' if language=='EN' else '中文逐页解释与时间分配')+'\n\nTotal: 8 minutes (480 seconds).\n'
    for j,c in enumerate(content,1):
        text+=f"\n## {j}. {c['title']} ({c['start']//60}:{c['start']%60:02d}, {c['seconds']} s)\n\n"+(c['speech'] if language=='EN' else c['zh'])+'\n\n'+('References: '+', '.join(c['references'])+'\n' if c['references'] else '')
    if language=='EN':
        text+='\n## References\n\n'+'\n'.join(f'- {a}: {b}. {c}' for a,b,c in refs)
        words=sum(len(c['speech'].split()) for c in content)
        text+=f'\n\nScript: {words} English words. Target pace: {words/8:.0f} words/minute. Timing is a rehearsal plan, not measured audio.\n'
    (HERE/f'SPEAKER_NOTES_{language}.md').write_text(text,encoding='utf-8')
print(sum(len(c['speech'].split()) for c in content),'spoken words')
