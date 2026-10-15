"""Report all three mechanisms, including unfavorable outcomes and policy costs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT
from extensions import CONFIG, OUT, MOD, safe_dump

POLICIES={'accuracy':'Accuracy objective','business':'Business: common threshold','business_group':'Business: group thresholds',
          'fair_10':'Fairness gap <=10 pp','fair_05':'Fairness gap <=5 pp','fair_02':'Fairness gap <=2 pp','fair_01':'Fairness gap <=1 pp'}
BRANCHES={'static':'Frozen model','selected_only':'Approved-only updates','explore_unweighted':'Exploration, unweighted','explore_ipw':'Exploration + IPW','random_reference':'Matched-size random reference*'}
COLORS={'static':'#6d8195','selected_only':'#c95245','explore_unweighted':'#b78b27','explore_ipw':'#188a7a','random_reference':'#765d9b'}
NAMES={'thin_frr':'薄弱组错拒率','thick_frr':'比较组错拒率','abs_frr_gap':'绝对错拒差距','approval_rate':'总体获批率','approved_risk':'获批风险'}


def bootstrap(y,g,left,right,seed=519):
    """Customer-paired pattern bootstrap, optionally resampling matched run seeds."""
    approvals=np.column_stack([v for pair in zip(left,right) for v in pair]).astype(np.int64)
    code=2*g+y
    for j in range(approvals.shape[1]): code=code | (approvals[:,j]<<(j+2))
    unique,counts=np.unique(code,return_counts=True)
    gy=unique&3; patterns=np.column_stack([(unique>>(j+2))&1 for j in range(approvals.shape[1])])
    thin=(gy==2).astype(float); thick=(gy==0).astype(float); bad=(gy%2).astype(float)
    rng=np.random.default_rng(seed)
    b=rng.multinomial(len(y),counts/len(y),size=CONFIG['bootstrap_repetitions'])
    def values(w):
        n=w@patterns
        f1=1-(w@(patterns*thin[:,None]))/(w@thin)[:,None]
        f0=1-(w@(patterns*thick[:,None]))/(w@thick)[:,None]
        risk=np.divide(w@(patterns*bad[:,None]),n,out=np.full(n.shape,np.nan),where=n>0)
        return {'thin_frr':f1,'thick_frr':f0,'abs_frr_gap':abs(f1-f0),'approval_rate':n/w.sum(axis=1)[:,None],'approved_risk':risk}
    base=values(counts[None,:]);samples=values(b)
    sampled_seeds=rng.integers(0,len(left),size=(len(b),len(left)))
    out=[]
    for key in NAMES:
        delta=samples[key][:,0::2]-samples[key][:,1::2]
        draws=np.take_along_axis(delta,sampled_seeds,axis=1).mean(axis=1)
        lo,hi=np.nanquantile(draws,[.025,.975])
        out.append({'metric':key,'delta':float((base[key][:,0::2]-base[key][:,1::2]).mean()),'ci_low':float(lo),'ci_high':float(hi)})
    return out


def intervals():
    records=[]
    full=pd.read_csv(OUT/'objective/full_test.csv')
    for left,right in [('business','accuracy'),('fair_10','business_group'),('fair_05','business_group'),('fair_02','business_group'),('fair_01','business_group'),('fair_02','business')]:
        for r in bootstrap(full.TARGET.to_numpy(),full.thin.to_numpy(),[full[left].to_numpy()],[full[right].to_numpy()]):
            records.append({'module':'objective','comparison':left+' - '+right,**r})
    biased=[pd.read_csv(OUT/f'objective/r50_s{s}_biased_test.csv') for s in CONFIG['feedback']['seeds']]
    for f in biased:
        if not np.array_equal(full.SK_ID_CURR,f.SK_ID_CURR):raise ValueError('Objective bootstrap alignment')
    for r in bootstrap(full.TARGET.to_numpy(),full.thin.to_numpy(),[f.fair_02.to_numpy() for f in biased],[f.business_group.to_numpy() for f in biased]):
        records.append({'module':'objective_biased','comparison':'fair_02 - business_group',**r})
    pop=pd.read_csv(OUT/'feedback/test_population.csv')
    for left,right in [('selected_only','static'),('explore_unweighted','selected_only'),('explore_ipw','selected_only'),('explore_ipw','explore_unweighted'),('selected_only','random_reference')]:
        sides=[]
        for branch in [left,right]:
            predictions=[]
            for seed in CONFIG['feedback']['seeds']:
                name=f's{seed}_initial' if branch=='static' else f's{seed}_{branch}_r5'
                with np.load(OUT/f'feedback/predictions/{name}.npz') as f: predictions.append(f['approved'])
            sides.append(predictions)
        for r in bootstrap(pop.TARGET.to_numpy(),pop.thin.to_numpy(),*sides):
            records.append({'module':'feedback','comparison':left+' - '+right,**r})
    df=pd.DataFrame(records)
    df.to_csv(OUT/'paired_intervals.csv',index=False)
    return df


def save(name):
    plt.savefig(ROOT/f'figures/{name}.png',dpi=180,bbox_inches='tight')
    plt.savefig(ROOT/f'figures/{name}.pdf',bbox_inches='tight')
    plt.close()


def figures(objectives,feedback,stream,ci):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    full=objectives[objectives.base_model=='full'].set_index('policy').reindex(POLICIES)
    fig,axs=plt.subplots(1,3,figsize=(13,4.6))
    palette=['#6d8195','#24475c','#5389a7','#dbb762','#c09a3c','#188a7a','#67b6a6']
    labels=['Accuracy','Business\ncommon','Business\ngroup','Fair\n10 pp','Fair\n5 pp','Fair\n2 pp','Fair\n1 pp']
    for ax,key,title in zip(axs,['abs_frr_gap','approval_rate','approved_risk'],['Absolute FRR gap (pp)','Approval rate (%)','Approved risk (%)']):
        ax.bar(np.arange(7),full[key]*100,color=palette)
        ax.set_xticks(np.arange(7),labels,fontsize=7,rotation=35,ha='right')
        ax.set(title=title);ax.grid(axis='y',alpha=.15)
        if key=='approved_risk':ax.axhline(5,color='#444',ls='--',lw=1)
        if key=='abs_frr_gap':ax.axhline(2,color='#188a7a',ls='--',lw=1)
    fig.suptitle('Same full-data risk model; different validation-selected objectives\nDashed lines are design targets, not test guarantees',fontsize=12)
    fig.tight_layout();save('06_objectives')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    fair=objectives[objectives.policy.str.startswith('fair')].copy()
    fair['tolerance']=fair.policy.str.split('_').str[1].astype(int)
    for is_full,label,color in [(True,'Full-data scores','#24475c'),(False,'Biased scores (5-seed mean)','#c95245')]:
        group=fair[(fair.base_model=='full')==is_full].groupby('tolerance')[['abs_frr_gap','approval_rate']].mean()
        axs[0].plot(group.index,group.abs_frr_gap*100,'o-',label=label,color=color)
        axs[1].plot(group.index,group.approval_rate*100,'o-',label=label,color=color)
    axs[0].plot([1,10],[1,10],'--',color='#888',label='Validation tolerance')
    axs[0].set(ylabel='Test absolute FRR gap (pp)');axs[1].set(ylabel='Test approval rate (%)')
    for ax in axs:ax.set(xlabel='Allowed validation FRR gap (pp)');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle('Fairness constraint sensitivity | same group-threshold search space')
    fig.tight_layout();save('07_objective_tradeoff')
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,key,title,mult in zip(axs.flat,['thin_frr','abs_frr_gap','auc','approved_risk'],['Thin-file false rejection (%)','Absolute FRR gap (pp)','ROC-AUC','Approved risk (%)'],[100,100,1,100]):
        for branch,label in BRANCHES.items():
            group=feedback[feedback.branch==branch].groupby('round')[key].agg(['mean','std'])
            ax.plot(group.index,group['mean']*mult,'o--' if branch=='random_reference' else 'o-',label=label,color=COLORS[branch],markersize=4)
            ax.fill_between(group.index,(group['mean']-group['std'])*mult,(group['mean']+group['std'])*mult,color=COLORS[branch],alpha=.12)
        ax.set(title=title,xlabel='Completed feedback rounds');ax.grid(alpha=.15)
        if key=='approved_risk':ax.axhline(5,color='#666',ls='--',lw=1)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Five-round historical replay | mean +/- SD across five seeds\n*Post-primary diagnostic: randomly reveal the same number of labels',fontsize=12)
    fig.tight_layout();save('08_feedback_trajectory')
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv')
    population=manifest[manifest.split=='train']
    true_share=((population.thin==1)&(population.TARGET==0)).mean()*100
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for ax,key,title in zip(axs,['thin_good_share','weighted_thin_good_share'],['Observed training representation','Representation after weighting']):
        for branch,label in BRANCHES.items():
            group=feedback[feedback.branch==branch].groupby('round')[key].mean()
            ax.plot(group.index,group*100,'o-',label=label,color=COLORS[branch],markersize=4)
        ax.axhline(true_share,color='#444',ls='--',label='Original training pool')
        ax.set(title=title,xlabel='Completed rounds',ylabel='Thin-file TARGET=0 share (%)');ax.grid(alpha=.15)
    axs[0].legend(fontsize=7);fig.tight_layout();save('09_feedback_representation')
    fig,axs=plt.subplots(1,2,figsize=(11,4.4))
    for branch,label in BRANCHES.items():
        if branch=='random_reference':continue
        group=stream[stream.branch==branch].groupby('round')[['actual_approved_risk','exploration_approved']].mean()
        axs[0].plot(group.index,group.actual_approved_risk*100,'o-',color=COLORS[branch],label=label)
        if branch.startswith('explore'):axs[1].plot(group.index,group.exploration_approved,'o-',color=COLORS[branch],label=label)
    axs[0].axhline(5,color='#444',ls='--');axs[0].set(ylabel='Actual approved risk incl. exploration (%)')
    axs[1].set(ylabel='Additional exploration approvals per round')
    for ax in axs:ax.set(xlabel='Arrival round');ax.grid(alpha=.15);ax.legend(fontsize=7)
    fig.suptitle('Exploration buys outcome coverage and changes risk exposure')
    fig.tight_layout();save('10_exploration_cost')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,comparison in zip(axs,['selected_only - static','explore_ipw - selected_only']):
        part=ci[(ci.module=='feedback')&(ci.comparison==comparison)].set_index('metric').reindex(NAMES)
        ax.errorbar(part.delta*100,np.arange(5),xerr=np.vstack([(part.delta-part.ci_low)*100,(part.ci_high-part.delta)*100]),fmt='o',capsize=4,color='#24475c')
        ax.axvline(0,color='#555');ax.set_yticks(np.arange(5),['Thin FRR','Thick FRR','Absolute gap','Approval rate','Approved risk'])
        ax.set(title=comparison.replace('_',' '),xlabel='Difference (pp), 95% paired interval');ax.grid(axis='x',alpha=.15)
    fig.suptitle('Final round effects, conditional on this historical replay')
    fig.tight_layout();save('11_feedback_effects')


def mdtable(df,cols):
    lines=['| '+' | '.join(label for _,label in cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in df.iterrows():lines.append('| '+' | '.join(str(r[key]) for key,_ in cols)+' |')
    return '\n'.join(lines)


def present(df,label_col,mapping):
    out=df.copy();out[label_col]=out[label_col].map(mapping)
    for key in ['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk','accuracy']:
        if key in out:out[key]=out[key].map(lambda x:f'{x*100:.2f}' if pd.notna(x) else '未定义')
    if 'auc' in out:out['auc']=out.auc.map(lambda x:f'{x:.4f}')
    return out


def main():
    done=json.loads((OUT/'completion.json').read_text())
    objective=pd.read_csv(OUT/'objective/metrics.csv')
    feedback=pd.read_csv(OUT/'feedback/test_metrics.csv')
    feedback=pd.concat([feedback,pd.read_csv(OUT/'feedback/diagnostic_metrics.csv')],ignore_index=True)
    diagnostic=json.loads((OUT/'feedback/diagnostic_completion.json').read_text())
    stream=pd.read_csv(OUT/'feedback/stream_metrics.csv')
    ci=intervals();figures(objective,feedback,stream,ci)
    objective.groupby('policy').mean(numeric_only=True).to_csv(OUT/'objective/summary_all_bases.csv')
    feedback.groupby(['branch','round']).agg({k:['mean','std'] for k in ['thin_frr','thick_frr','abs_frr_gap','auc','approval_rate','approved_risk','thin_good_share','weighted_thin_good_share']}).to_csv(OUT/'feedback/summary.csv')
    full=objective[objective.base_model=='full'].set_index('policy')
    fair=full.loc['fair_02'];business=full.loc['business_group']
    strict=full.loc['fair_01']
    biased_summary=objective[objective.base_model!='full'].groupby('policy').mean(numeric_only=True)
    final=feedback[feedback['round']==5].groupby('branch').mean(numeric_only=True)
    def effect(comparison,metric='thin_frr'):
        return ci[(ci.module=='feedback')&(ci.comparison==comparison)&(ci.metric==metric)].iloc[0]
    change=effect('selected_only - static');repair=effect('explore_ipw - selected_only')
    selection_effect=effect('selected_only - random_reference')
    if change.ci_low>0:
        feedback_claim='只用获批反馈更新，相对静态模型使薄弱组错拒率上升，支持此设定下的放大效应。'
    elif change.ci_high<0:
        feedback_claim='只用获批反馈更新，相对静态模型的薄弱组错拒率下降；本设定没有证实该指标的偏见放大。应继续查看群体差距、评分质量及风险，不能把反馈循环描述成必然恶化。'
    else:
        feedback_claim='只用获批反馈更新与静态模型的薄弱组错拒率差值区间包含 0，未得到稳定放大的证据。'
    repair_claim=('探索加权相对只用获批反馈的薄弱组错拒率更低，且在固定划分的条件 bootstrap 中，差值区间不跨零；这不是跨业务场景的显著性保证。' if repair.ci_high<0 else
                  '探索加权相对只用获批反馈未稳定降低薄弱组错拒率，不能把它包装为全面更优。')
    outcomes=stream[stream.branch=='explore_ipw']
    weighted_risk=np.average(outcomes.actual_approved_risk,weights=outcomes.actual_approved)
    extra=outcomes.groupby('seed').exploration_approved.sum().mean()
    objective_table=present(full.reset_index(),'policy',{'accuracy':'准确率目标','business':'业务：统一阈值','business_group':'业务：分组阈值','fair_10':'公平：差距≤10pp','fair_05':'公平：差距≤5pp','fair_02':'公平：差距≤2pp','fair_01':'公平：差距≤1pp'})
    feedback_table=present(final.reindex(BRANCHES).reset_index(),'branch',{'static':'静态不更新','selected_only':'仅获批反馈','explore_unweighted':'随机探索，等权','explore_ipw':'随机探索＋逆概率加权','random_reference':'同规模随机反馈（追加诊断）'})
    biased_table=present(biased_summary.reset_index(),'policy',{'accuracy':'准确率目标','business':'业务：统一阈值','business_group':'业务：分组阈值','fair_10':'公平：差距≤10pp','fair_05':'公平：差距≤5pp','fair_02':'公平：差距≤2pp','fair_01':'公平：差距≤1pp'})
    effect_table=ci[ci.module=='feedback'].copy()
    effect_table['metric']=effect_table.metric.map(NAMES)
    effect_table['estimate']=effect_table.apply(lambda r:f'{r.delta*100:+.2f} [{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]',axis=1)
    test_fair_breach=int(objective[objective.policy.str.startswith('fair')].test_gap_exceeds_tolerance.sum())
    test_risk_breach=int(objective[objective.policy!='accuracy'].test_risk_exceeds_budget.sum())
    core=pd.read_csv(ROOT/'results/metrics.csv')
    core_main=core[(core.retention==.5)&(core.budget==.05)].groupby('condition').mean(numeric_only=True)
    a_delta=(core_main.loc['corrected','thin_frr']-core_main.loc['biased','thin_frr'])*100
    text=f'''# 完整实验结果报告：数据、目标与反馈如何影响信贷机会

## 一、三问三答

| 问题 | 控制什么 | 实验回答 |
|---|---|---|
| 数据怎么来？ | 选择性采样与同规模随机删减 | 主实验加权纠正使薄弱组错拒率变化 {a_delta:+.2f} 个百分点，恢复了大部分受控偏差造成的额外错拒 |
| 系统优化什么？ | 固定风险评分，改变审批目标及约束 | 完整模型的 2pp 约束未改变策略；五个偏置模型上差距由 {biased_summary.loc['business_group','abs_frr_gap']*100:.2f}pp 降至 {biased_summary.loc['fair_02','abs_frr_gap']*100:.2f}pp，伴随获批率变化 |
| 系统如何更新？ | 只看获批反馈与探索/加权，多轮回放 | 未证实相对初始状态持续放大；但末轮仅获批反馈比同规模随机反馈多 {selection_effect.delta*100:.2f}pp 薄弱组错拒，探索加权相对仅获批更新变化 {repair.delta*100:+.2f}pp |

**核心解释：**数据代表性、决策目标和标签可见性是不同机制。不能用一个公平指标的下降断言所有群体都受益，也不能把反馈循环的存在等同于偏差必然放大。

本报告涵盖 31 个原始采样对照模型、42 个审批策略对照、80 个预设反馈模拟模型及 25 个追加诊断模型。全部在本机 CPU、4 线程运行。新增主模块耗时 {done['seconds_this_invocation']:.1f} 秒，追加诊断 {diagnostic['seconds']:.1f} 秒（不含 bootstrap、制图和核验）。

## 二、共同背景与数据边界

使用 Home Credit 的 307,511 条有标签申请和 1,716,428 条征信记录，通过固定版本的第三方 GitHub 镜像获取，完整来源与哈希见 `data/raw/source.json`。镜像结构、标签数量和键已核验，但没有官方逐值比对。AWS 官方项目作为流程参考，本项目没有运行 SageMaker 云服务，也不代表银行生产审批模型。

薄弱组定义为申请前可观察征信记录 0—1 条，比较组至少 2 条；零记录可能来自覆盖不足。TARGET=1 表示前 Y 期中至少一期逾期超过 X 天，X/Y 未公开；“错拒”指 TARGET=0 客户被拒，并非保证这些客户永远不违约。

训练/参考/校准/验证/测试划分沿用 50/10/10/10/20%，各集合贷款记录 ID（SK_ID_CURR）不重叠；字段字典将其定义为贷款 ID，不能据此保证自然人或家庭不跨集合。模块 B 使用已训练模型的固定分数；模块 C 仅把原训练池拆成模拟初始历史和未来到达批次。原测试集在模块 A 已被查看，因此 B/C 是固定追加协议的后续机制分析，**不宣称全新外部盲测**。

### 审查后明确的结论边界

- 本项目采用相对于 TARGET 的群体机会准则；征信记录较少不自动等于受保护身份，FRR 差距本身不能证明不公平歧视。
- 模块 A 同时改变组内标签比例和群体人数，尚未分别识别纯粹人数不足与标签结构失真的贡献；纠正依赖代表性有标签参考集。
- 模块 C 始终有独立验证标签支持阈值选择，且假定每批结果已成熟后再更新，不能代表无外部验证支持或存在标签延迟的真实闭环。
- 这些范围限制及后续所需对照详见 [原则与逻辑审查](AUDIT_REPORT.md)。

## 三、模块 A：数据怎么来——采样不平衡与训练数据偏见

已有实验详细数值见 [模块 A 报告](RESULTS_REPORT.md)。主要设计是：仅删减薄弱组 TARGET=0，保留 50% / 25%，与相同人数随机删减对照，真实标签不变；用独立参考集的组别×标签联合比例估计纠正权重。

- 50% 保留率、5% 验证风险预算下，偏置模型薄弱组错拒率 {core_main.loc['biased','thin_frr']*100:.2f}%，纠正后 {core_main.loc['corrected','thin_frr']*100:.2f}%。
- 获批者风险从 {core_main.loc['biased','approved_risk']*100:.2f}% 变为 {core_main.loc['corrected','approved_risk']*100:.2f}%；比较组错拒也有变化，不能只谈受益群体。
- 数据偏见由实验人为设定；它说明这种机制可以造成不利差别，不能证明 Home Credit 实际按此机制采样。

![采样机制](figures/01_mechanism.png)

## 四、模块 B：系统优化什么——固定模型，不同审批目标

### 4.1 设计与公式

固定预测概率 p，批准规则为 p≤t。准确率目标在验证集最小化误分类次数；业务目标在验证获批者风险≤5%下最大化批准人数。公平目标进一步约束两组 TARGET=0 错拒率绝对差距≤ε。

`max sum(approved)`，约束 `sum(TARGET * approved) <= 0.05 * sum(approved)`，公平策略另有 `abs(FRR_thin - FRR_thick) <= epsilon`。

统一阈值业务策略之外，加入**相同双阈值搜索空间但没有公平约束**的 `business_group`，因此公平目标的主比较是 fair_02−business_group。否则“分组阈值更灵活”与“加入公平约束”会混在一起。两种策略都在批准人数和风险完全相同时优先选择更小差距，因此此处比较的是增加公平硬约束的作用。每组网格最多约 514 个阈值，包含统一业务阈值；只宣称网格内最优，非连续全局最优。

这是审批后处理的目标函数实验，**没有改写 XGBoost 训练损失**，也不调用 Fairlearn 的随机化 ThresholdOptimizer。分组公平策略在决策时需要组别信息，是研究对照，不声称可直接上线。阈值均由验证集选择，之后固定用于测试。

### 4.2 完整数据模型上的测试结果

所有百分率列的单位为 %；差距列为百分点（pp）。

{mdtable(objective_table,[('policy','策略'),('accuracy','分类准确率'),('auc','AUC'),('thin_frr','薄弱组错拒'),('thick_frr','比较组错拒'),('abs_frr_gap','绝对差距'),('approval_rate','获批率'),('approved_risk','获批风险')])}

注意所有策略 AUC 相同，因为底层风险排序相同；改变的是决策规则。高准确率本身没有表达业务风险预算或群体机会目标。公平约束并不保证风险、准确率、获批机会同时更好，必须结合整张表判断。

**这里有两个不能省略的发现：**准确率目标几乎批准全部申请，错拒差距接近零，但获批风险接近 8%，说明小差距本身不足以定义好策略。完整数据模型上，分组业务最优解已满足 2pp 验证容差，因此加上 2pp 约束后阈值和结果不变；不能把统一阈值到分组阈值的改善归功于公平约束。预设的更严格 1pp 约束才进一步把测试差距从 {business.abs_frr_gap*100:.2f}pp 降到 {strict.abs_frr_gap*100:.2f}pp，获批率从 {business.approval_rate*100:.2f}% 变为 {strict.approval_rate*100:.2f}%。

主公平策略的验证容差是 2pp，测试差距为 {fair.abs_frr_gap*100:.2f}pp。全部六个底层模型、四档公平约束共 24 个公平策略中，有 {test_fair_breach} 个测试差距超过各自验证容差；36 个带风险预算的策略中，有 {test_risk_breach} 个测试风险超过 5%。全部结果都保留；本次达标也不意味着未来一定达标。

### 4.3 偏置模型敏感性分析（五个种子均值）

{mdtable(biased_table,[('policy','策略'),('thin_frr','薄弱组错拒%'),('thick_frr','比较组错拒%'),('abs_frr_gap','差距pp'),('approval_rate','获批率%'),('approved_risk','获批风险%')])}

在这些偏置模型上，2pp 公平约束相对同策略空间的业务对照更有约束作用。其获批率从 {biased_summary.loc['business_group','approval_rate']*100:.2f}% 变为 {biased_summary.loc['fair_02','approval_rate']*100:.2f}%。这说明公平约束的作用依赖已有评分与群体风险分布，并非在所有模型上都会产生同样幅度的改变。

敏感性分析使用五个偏置模型，全部 42 行策略结果在 `results/extensions/objective/metrics.csv`。公平相对业务策略的配对区间在 `results/extensions/paired_intervals.csv`。

![审批目标](figures/06_objectives.png)

![约束敏感性](figures/07_objective_tradeoff.png)

## 五、模块 C：系统如何更新——五轮反馈回放

### 5.1 模拟时间与标签可见性

每个种子把原训练池随机分成初始 20% 及五批新申请。初始历史仍对薄弱组 TARGET=0 保留 50%，四条路径共享同一初始偏置模型：静态不更新、只用获批反馈更新、10% 被拒客户随机探索后等权更新、同样探索后逆概率加权更新。

每轮用上轮模型先审批，只有常规获批和探索批准客户的标签才可见，轮末重训服务下一轮。这里假定该批还款观察窗口已经结束、标签已经成熟；模拟轮次不能对应未经验证的现实日历间隔。拒绝且未探索的结果在观测台账中留空，不能进入训练。探索获批的可见概率是 0.1，常规获批为 1；IPW 使用对应逆概率 10 或 1，归一化均值为 1。初始历史不做回溯修复，因此初始训练偏差不能被宣称一次消除。

所有路径都能使用固定独立校准和验证数据，逐轮校准和选择 5% 风险阈值；测试集从不用于调阈值。各路径对新批次的选择会不同，探索等权与加权不是在每一轮强制共享完全相同的训练记录，比较的是完整动态策略。每种子新增 16 个模型，共 80 个，固定 200 棵树、depth=5，不通过测试轨迹选轮次或调参。

这是历史数据回放：随机批次不是实际日期，还款结果沿用历史标签，假设不同审批路径不会改变个人潜在结果。真实部署时拒贷者结果不可知，探索意味着额外放贷才能获得反馈。该模拟只研究样本选择反馈，不模拟负债行为变化、宏观漂移或福利的长期因果效应。

**追加诊断的时间顺序：**看到预设四路径的轨迹后，我们发现“仅获批更新−静态”同时包含样本增长和选择机制两种变化，因此追加同规模随机反馈参照（详见 FEEDBACK_DIAGNOSTIC.md）。每轮随机获得与仅获批路径相同数量的标签，累计训练人数完全一致，初始模型和其他条件一致。它是理想化随机审计，不是可免费获得拒贷结果的部署策略；这是后续诊断，不冒充事前登记的主要检验。

### 5.2 末轮独立测试结果（五个种子均值）

{mdtable(feedback_table,[('branch','路径'),('auc','AUC'),('thin_frr','薄弱组错拒%'),('thick_frr','比较组错拒%'),('abs_frr_gap','差距pp'),('approval_rate','获批率%'),('approved_risk','获批风险%')])}

{feedback_claim}

{repair_claim}

探索加权也有代价：末轮 AUC 从仅获批更新的 {final.loc['selected_only','auc']:.4f} 变为 {final.loc['explore_ipw','auc']:.4f}，总体获批率从 {final.loc['selected_only','approval_rate']*100:.2f}% 变为 {final.loc['explore_ipw','approval_rate']*100:.2f}%，比较组错拒率从 {final.loc['selected_only','thick_frr']*100:.2f}% 变为 {final.loc['explore_ipw','thick_frr']*100:.2f}%。因此它没有同时优化所有指标。

薄弱组错拒变化：仅获批更新−静态 = {change.delta*100:+.2f}pp，近似 95% 区间 [{change.ci_low*100:+.2f}, {change.ci_high*100:+.2f}]；探索加权−仅获批更新 = {repair.delta*100:+.2f}pp，区间 [{repair.ci_low*100:+.2f}, {repair.ci_high*100:+.2f}]。

**同等标签数量的选择机制诊断：**末轮仅获批更新−随机反馈参照的薄弱组错拒差值为 {selection_effect.delta*100:+.2f}pp，区间 [{selection_effect.ci_low*100:+.2f}, {selection_effect.ci_high*100:+.2f}]。这项比较更接近隔离标签选择方式；它与“相比初始状态是否改善”回答不同问题。不得因初始到末轮改善，就断言选择性反馈没有代价。

不能只比较末轮绝对差距：若两组同时更难获批，差距也可能缩小。表中同时展示两组错拒、总体批准和风险；轨迹保留所有轮次，而非选择最大改善的轮次。

![多轮轨迹](figures/08_feedback_trajectory.png)

![样本代表性](figures/09_feedback_representation.png)

### 5.3 探索并非没有代价

探索加权路径每个模拟种子五轮平均额外批准 {extra:.1f} 个原本被拒的申请。把探索批准也计入后，五轮实际批准的加权汇总观察风险为 **{weighted_risk*100:.2f}%**。这与上表“常规审批策略在独立测试上的风险”不是同一口径；5% 的常规阈值预算不约束探索带来的额外风险。

记录 `stream_metrics.csv` 中的每轮风险、探索人数及薄弱组标签覆盖；训练权重有效样本量、最大权重在模型决策文件和 `test_metrics.csv` 中。IPW 的分布纠正与大权重导致的方差同时存在，不能只展示代表性改善。

![探索成本](figures/10_exploration_cost.png)

### 5.4 末轮配对差值与不确定性

{mdtable(effect_table,[('comparison','比较'),('metric','指标'),('estimate','差值及95%区间（pp）')])}

![配对差值](figures/11_feedback_effects.png)

配对 bootstrap 1000 次，同一测试贷款记录在所有模型中同步抽样，反馈比较还重采样五个模拟种子。区间条件于已固定的人群划分、校准/验证集及模拟参数，未覆盖真实时间变化或拒贷反事实的不确定性；多个辅助指标未作多重检验校正。

## 六、汇报结论怎么写

1. **数据来源机制会改变模型学到的风险关系。**选择性缺失某群体的正常还款案例，可增加该群体错拒；匹配机制的加权能够恢复部分表现。
2. **预测准确率、业务约束与公平目标表达不同的价值取舍。**同一评分模型可以产生不同审批结果；引入公平约束需要同时报告风险和各群体代价。
3. **审批决定会改变下一轮可以观察到的数据。**在独立验证集持续支持阈值选择、各批标签已成熟的设定下，本模拟没有发现薄弱组错拒随更新持续放大，但后续同规模诊断显示选择性反馈仍比随机反馈产生更多错拒。因此“模型比以前好”不代表反馈选择没有代价。探索和纠正可以改善信息覆盖，但有风险敞口、预测表现和群体机会取舍。

建议英文 PPT 使用 01（数据）、06（目标）、08（反馈）作为三张主图，10（探索成本）和11（置信区间）备答。可在云计算架构页展示：数据处理→训练→偏差审计→策略选择→反馈采集→下一轮训练；标注本次是 AWS-inspired 本地 CPU 验证，没有测量云端扩展性能。

## 七、复现、核验与资料

- 完整协议：`EXTENSION_PROTOCOL.md`；固定参数：`extensions_config.json`。
- 执行 `./run_complete.ps1`；新增程序 `src/extensions.py`，报告 `src/complete_report.py`，独立核验 `src/verify_extensions.py`。
- 核验记录：`results/extensions/verification.json`，检查阈值约束、测试指标重算、模拟新批次不重叠、先决策后观测、未观察标签不入训练、权重来源及静态对照一致性。
- 模块 A 原结果保持不变，完整复现环境见 README 和 requirements-lock.txt。
- [AWS 官方流程](https://github.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection)
- [Fairlearn 官方信贷决策示例](https://fairlearn.org/v0.12/auto_examples/plot_credit_loan_decisions.html)
- [Selective Labels, KDD 2017](https://www.cs.cornell.edu/home/kleinber/kdd17-selective.pdf)
- [Delayed Impact of Fair Machine Learning, ICML 2018](https://proceedings.mlr.press/v80/liu18c.html)
'''
    (ROOT/'COMPLETE_RESULTS_REPORT.md').write_text(text,encoding='utf-8')
    slides=f'''# 英文汇报可直接使用的实验摘要

## Slide 1 — Where does the data come from?

- Real Home Credit data; controlled selective under-sampling.
- Thin-file borrowers: 0–1 observable bureau records.
- Reweighting reduced thin-file false rejections by {-a_delta:.2f} percentage points in the main scenario.
- This demonstrates a possible mechanism, not misconduct by the data provider.

Figure: `figures/01_mechanism.png`

## Slide 2 — What does the system optimize?

- The risk scores are fixed; approval objectives change.
- Compare accuracy, risk-constrained approvals, and approvals with a fairness constraint.
- For full-data scores, the 2-pp constraint is non-binding: it changes no decisions.
- For biased scores, the same constraint reduces the mean gap from {biased_summary.loc['business_group','abs_frr_gap']*100:.2f} to {biased_summary.loc['fair_02','abs_frr_gap']*100:.2f} pp; approvals change from {biased_summary.loc['business_group','approval_rate']*100:.2f}% to {biased_summary.loc['fair_02','approval_rate']*100:.2f}%.
- Validation constraints are not test guarantees; report both groups and risk.

Figure: `figures/06_objectives.png`

## Slide 3 — How does the system update?

- Five rounds of selective-label historical replay; five simulation seeds.
- All branches retain independent labeled validation support; each batch is assumed to mature before the next update.
- Approved-only updates vs. a frozen model: thin-file FRR changes by {change.delta*100:+.2f} pp.
- Exploration + IPW vs. approved-only updates: {repair.delta*100:+.2f} pp.
- Post-primary diagnostic: approved-only vs. matched-size random feedback, {selection_effect.delta*100:+.2f} pp.
- Under these assumptions, no amplification was observed from the initial state; selective feedback still trails the matched-size random reference.

Figure: `figures/08_feedback_trajectory.png`

## Slide 4 — What does mitigation cost?

- Exploration reveals more outcomes by approving some otherwise rejected applicants.
- Actual risk including exploration: {weighted_risk*100:.2f}% in this replay.
- Weighting can improve representation but increases estimation variance.
- Local CPU implementation of an AWS-inspired workflow; no production or causal claims.

Figure: `figures/10_exploration_cost.png`
'''
    (ROOT/'PRESENTATION_NOTES.md').write_text(slides,encoding='utf-8')
    print('Complete report, presentation notes, paired intervals, and six figure pairs generated.')


if __name__=='__main__':main()
