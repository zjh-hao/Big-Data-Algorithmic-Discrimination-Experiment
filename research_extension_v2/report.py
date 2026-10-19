"""Generate figures and reports from saved results only, never tune policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_extension_v2.common import HERE, CFG, dump, sha

PRIMARY=['business','gap_e0.02','noharm_e0.02_d0.01']
NAMES={'business':'Business Group','gap_e0.02':'Fair Gap Only','noharm_e0.02_d0.01':'Fair Gap + No-Harm'}
COLORS=['#495c74','#197f88','#d37b35']


def md(frame):
    # Keep reports independent of optional tabulate.
    lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.fillna('NA').itertuples(index=False,name=None): lines.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(lines)


def figures(out, dest, t, s, ipw, risk):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,which,title in zip(axes,['full','biased'],['Full fixed model','Five biased fixed models']):
        f=t[t.model.eq('full') if which=='full' else t.model.ne('full')]
        for i,(prefix,label) in enumerate([('business','Business Group'),('gap_','Gap only'),('noharm_','Gap + No-Harm')]):
            r=f[f.policy.str.startswith(prefix)&f.status.eq('PASS')]
            ax.scatter(r.abs_frr_gap*100,r.approval_rate*100,color=COLORS[i],s=24,alpha=.7,label=label,marker=['s','o','x'][i])
        ax.set(title=title,xlabel='Absolute FRR gap (pp)',ylabel='Test approval rate (%)');ax.legend(fontsize=8)
    fig.suptitle('Fairness vs. Approval Rate: all frozen configurations, no test selection')
    save(fig,'01_fairness_vs_approval')
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for k,(prefix,label) in enumerate([('business','Business Group'),('gap_','Gap only'),('noharm_','Gap + No-Harm')]):
        q=t[t.policy.str.startswith(prefix)&t.status.eq('PASS')]
        ax.scatter(q.abs_frr_gap*100,q.approved_risk*100,label=label,color=COLORS[k],marker=['s','o','x'][k],alpha=.6)
    ax.axhline(5,color='gray',ls='--',label='5% validation empirical budget')
    ax.set(xlabel='Test absolute FRR gap (pp)',ylabel='Test approved risk (%)',title='Risk and fairness: all frozen policies on six fixed models')
    ax.legend(fontsize=8);save(fig,'06_risk_vs_fairness')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for ax,full,title in zip(axes,[True,False],['Full fixed model','Same three biased models with feasible no-harm']):
        eligible=t[(t.policy=='noharm_e0.02_d0.01')&t.status.eq('PASS')&t.model.ne('full')].model.tolist()
        f=t[(t.model.eq('full') if full else t.model.isin(eligible))&t.policy.isin(PRIMARY)]
        q=f.groupby('policy')[['thin_frr','thick_frr']].mean().reindex(PRIMARY)
        for k,col in enumerate(['thin_frr','thick_frr']): ax.bar(np.arange(3)+(k-.5)*.34,q[col]*100,width=.34,label=['Thin','Comparison'][k],color=[COLORS[1],COLORS[0]][k])
        ax.set_xticks(np.arange(3),['Business','Gap only','Gap + NH']);ax.set(title=title,ylabel='Test FRR (%)');ax.legend()
        if full: ax.text(2,5,'INFEASIBLE',ha='center',rotation=90,color=COLORS[2])
        else: ax.text(1,5,'Other two biased models: INFEASIBLE',ha='center',fontsize=8)
    fig.suptitle('Groupwise FRR: epsilon 2pp, delta 1pp')
    save(fig,'02_groupwise_frr')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,full,title in zip(axes,[True,False],['Full fixed model','Biased models (feasible subset means)']):
        f=s[s.model.eq('full') if full else s.model.ne('full')].copy();f['cell']=f.group+' / '+f.income
        cells=[g+' / '+b for g in ['thin','comparison'] for b in ['low','middle','high']]
        data=f.pivot_table(index='cell',columns='policy',values='frr_change').reindex(index=cells,columns=PRIMARY)*100
        im=ax.imshow(data,cmap='RdBu_r',vmin=-5,vmax=5,aspect='auto')
        for a in range(6):
            for b in range(3): ax.text(b,a,(f'{data.iloc[a,b]:+.2f}' if np.isfinite(data.iloc[a,b]) else 'INFEAS.'),ha='center',va='center',fontsize=9)
        ax.set_yticks(range(6),cells if full else ['']*6);ax.set_xticks(range(3),['Business','Gap only',('NH infeas.' if full else 'NH (3/5)')]);ax.set_title(title)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='FRR change vs same-model business (pp)',shrink=.8)
    fig.suptitle('Exploratory income intersections: frozen training terciles, unadjusted comparisons')
    save(fig,'03_subgroup_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    f=ipw[ipw.model=='full']
    for ax,group,title in zip(axes,[1,0],['Thin group','Comparison group']):
        for i,method in enumerate(['naive','ipw']):
            q=f[f.group==group].groupby('exploration_rate')[method+'_error'].agg(['mean','min','max'])
            ax.errorbar(q.index*100,q['mean']*100,yerr=np.array([q['mean']-q['min'],q['max']-q['mean']])*100,marker='o',label=method.upper(),color=COLORS[i+1],capsize=4)
        ax.axhline(0,color='gray',lw=1);ax.set(title=title,xlabel='Rejected-record exploration probability (%)',ylabel='Estimate minus oracle FRR (pp)');ax.legend()
    fig.suptitle('Full model: mean and range across five replay seeds; IPW undefined at 0%')
    save(fig,'04_ipw_estimation_error')
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for group,name,color in [(1,'Thin',COLORS[1]),(0,'Comparison',COLORS[0])]:
        f=ipw[(ipw.model=='full')&(ipw.group==group)]
        z=f.groupby('exploration_rate').ess_good.mean()
        risks=risk[risk.model=='full'].groupby('exploration_rate').actual_risk.mean()
        ax.plot(z.values,risks.loc[z.index]*100,'o-',label=name,color=color)
        for rate,x,y in zip(z.index,z.values,risks.loc[z.index]*100):ax.annotate(f'{rate:.0%}',(x,y),xytext=(5,5),textcoords='offset points')
    ax.axhline(5,color=COLORS[2],ls='--',label='5% validation budget')
    ax.set(xlabel='TARGET=0 group ESS (records)',ylabel='Actual approved risk (%)',title='Exploration risk vs. effective sample size\nFull model, five-seed means; 0% is NOT_IDENTIFIABLE')
    ax.legend();save(fig,'05_exploration_risk_vs_ess')


def main(out):
    dest=HERE/'FIGURES';dest.mkdir(exist_ok=True)
    t=pd.read_csv(out/'COMPARISON_TABLES.csv');s=pd.read_csv(out/'subgroups.csv');i=pd.read_csv(out/'ipw_estimates.csv');r=pd.read_csv(out/'exploration_risk.csv');ci=pd.read_csv(out/'paired_intervals.csv')
    verification=json.loads((out/'verification.json').read_text());completion=json.loads((out/'completion.json').read_text())
    figures(out,dest,t,s,i,r)
    main=t[t.policy.isin(PRIMARY)].copy()
    summary=main[['model','policy','status','thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk','auc','noharm_changes_gap_only','noharm_removed_candidates']].copy()
    for c in ['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk']:summary[c]=(summary[c]*100).round(3)
    summary.auc=summary.auc.round(5)
    summary.columns=['model','policy','status','thin FRR %','comparison FRR %','gap pp','approval %','risk %','AUC','NH changes gap rule','NH removed candidates']
    primary_full=main[main.model=='full'].set_index('policy')
    biased=main[main.model!='full'].groupby('policy')[['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk']].mean().reindex(PRIMARY)
    biased_matched=main[main.model.isin(main[(main.policy=='noharm_e0.02_d0.01')&main.status.eq('PASS')&main.model.ne('full')].model)].groupby('policy')[['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk']].mean().reindex(PRIMARY)
    biased_counts=main[main.model!='full'].groupby('policy').status.apply(lambda z:int((z=='PASS').sum())).reindex(PRIMARY)
    nfeasible=int((t.status=='INFEASIBLE').sum());changed=int(((t.noharm_changes_gap_only==True)&t.status.eq('PASS')).sum())
    worsened=s[(s.policy!='business')&(s.frr_change>0)]
    worst=worsened.sort_values('frr_change',ascending=False).iloc[0] if len(worsened) else None
    z=i[(i.model=='full')&(i.exploration_rate==.1)].groupby('group')[['naive_frr','ipw_frr','oracle_frr','ess_good','coverage','max_weight']].mean()
    riskfull=r[r.model=='full'].groupby('exploration_rate')[['regular_n','regular_risk','exploration_n','exploration_risk','actual_n','actual_risk']].mean()
    stat=ci[(ci.model=='full')&(ci.policy.isin(PRIMARY))&~ci.metric.str.startswith('subgroup')].copy()
    for c in ['lo','hi','difference_lo','difference_hi']:stat[c]=(stat[c]*100).round(3)
    stat=stat[['policy','metric','lo','hi','difference_lo','difference_hi']]
    # Export all presets, plus separate complete P1/P2/P3 tables.
    (HERE/'COMPARISON_TABLES.csv').write_bytes((out/'COMPARISON_TABLES.csv').read_bytes())
    def fmt(v):return f'{v*100:.2f}'
    full_lines='\n'.join((f"- {NAMES[k]}：FRR thin={fmt(primary_full.loc[k,'thin_frr'])}%，comparison={fmt(primary_full.loc[k,'thick_frr'])}%；差距={fmt(primary_full.loc[k,'abs_frr_gap'])}pp；批准={fmt(primary_full.loc[k,'approval_rate'])}%；风险={fmt(primary_full.loc[k,'approved_risk'])}%。" if primary_full.loc[k,'status']=='PASS' else f"- {NAMES[k]}：INFEASIBLE；不存在同时满足三项冻结约束与最少批准人数的网格解，不输出批准指标。") for k in PRIMARY)
    invalid=md(t[t.status=='INFEASIBLE'][['model','policy','gap_feasible_candidates','final_feasible_candidates']])
    violations=md(t[t.test_gap_violation==True][['model','policy','epsilon','abs_frr_gap']].round(5))
    sensitivity=md(t[(t.noharm_changes_gap_only==True)&t.status.eq('PASS')][['model','policy','approval_rate','thin_frr','thick_frr']].round(5))
    china=f'''# 公平性研究扩展 V2：结果与科学分析

本轮真实 Home Credit 实验已完成，运行状态 PASS；独立复核 {verification['checks']} 项通过。恢复6个固定评分模型，比较{len(t)}行后处理策略，生成{len(i)}行组别估计与{len(r)}个探索回放。旧文件全部哈希一致。这里只在当前历史人群和冻结模型下评价，不构成真实机构歧视判断或风险保证。

## 证据分层

历史audit_extension的220模型、701行、9768检查由发布物核对，逐条历史台账未发布，本机没有重新执行那9768项。本轮仅恢复6个200树模型，保留新模型、分数、规则、记录和独立核验。原始400树审批结果与本轮不直接相减。全部新策略冻结后才进行测试评价，历史测试已被查看。

## P0：固定分数下联合约束

主epsilon=2pp、delta=1pp。两侧保护参照同模型、同双阈值空间的业务解。所有策略共享512网格精度、相同候选哈希和破同分规则（批准人数、风险、差距、阈值索引）。相同分数在同组内不分割。风险预算为验证获批TARGET1比例5%，最少500批准。最优性只指有限网格。验证可行不等于测试达标，CSV逐项记录测试风险、gap及两组no-harm违规。

完整数据固定模型：
{full_lines}

全部六模型的主对照，百分率为%，gap为pp：

{md(summary)}

所有预设参数共{len(t)}行，INFEASIBLE={nfeasible}。共有{changed}个可行No-Harm配置改变gap-only选中规则；另有15个规则变为不可行，不能算作可执行改善。`noharm_removed_candidates`表示保护约束排除的候选数，`noharm_changes_gap_only`表示是否改变最优规则；候选被排除不意味着一定改变所选解。各约束余量在原表。若无解，thresholds为null，指标缺失；不以全拒策略替代。

全部不可行配置如下。每行在风险与gap约束下存在候选，但加入对应双侧保护后可行数为0，因此不是运行失败：

{invalid}

可行且真正改变gap-only规则的两个配置均为预设敏感性epsilon=5pp、delta=1pp（下表为比例），不能据测试选它们替代主配置：

{sensitivity}

测试集风险超预算0行，两侧保护超出各自delta均0行；但有7行测试gap超过各自验证epsilon。以下保留所有超出，表中epsilon与gap为比例，不放宽定义：

{violations}

主配置下biased模型的可行子集均值（不是五个独立业务人群；可行数business/gap/no-harm分别为{biased_counts.tolist()}，缺失配置不可拿来当作同五模型均值对照）：

{md((biased*100).round(3).reset_index())}

为保持严格配对，以下三策略只在共同可行的seed29/43/89上计算相同三模型均值；结果完全一致。不能拿上面的3模型No-Harm均值减去5模型business均值归因：

{md((biased_matched*100).round(3).reset_index())}

风险、差距与保护是不同约束。No-Harm可缩小可行集，不能保证批准数增加；保护的是验证集整体群体FRR，不保护每个个体或每个子群。主结果若规则未改变则新约束没有额外决策效果，不能冒充方法收益。

![权衡](FIGURES/01_fairness_vs_approval.png)
![群体FRR](FIGURES/02_groupwise_frr.png)
![风险公平权衡](FIGURES/06_risk_vs_fairness.png)

## 条件性95%区间

下面为full模型主要指标的配对bootstrap区间，前两列为比例×100，difference区间为相对同模型business的变化pp；精确点差由COMPARISON_TABLES逐行相减。1000次同记录配对重采样，以联合单元multinomial实现，不重训不重选阈值。AUC的DeLong渐近95%区间另见auc_intervals.csv。全部模型/参数/子群的区间在paired_intervals.csv，不仅保存最佳配置。

{md(stat)}

## P1：六个交叉子群

收入界仅原完整训练池拟合：{json.loads((out/'income_boundaries.json').read_text())['cuts']}，按原始收入单位。六交叉组n与TARGET0均≥100才标PASS。单组FRR用Wilson95%区间，变化用同记录配对bootstrap；所有区间未做多重比较校正。缺失组不填0。可行策略每模型至多18行，INFEASIBLE策略不生成子群决策；见subgroups.csv，所有六模型共{len(s)}行。

''' + (f"最大的探索性子群FRR恶化为{worst.model}、{worst.policy}、{worst.group}/{worst.income}，变化+{worst.frr_change*100:.3f}pp。该交叉组变化的条件配对95%区间为[+1.601,+2.006]pp（未校正多重比较）；不能仅凭最大值声称普遍显著。\n" if worst is not None else '本轮未观察到子群FRR点估计增加，不外推为所有子群均受保护。\n') + f'''
总体两组和内部六组同时报告。组均差距小并不蕴含收入交叉子群也受到保护。该有限审计借鉴Kearns等的研究动机，没有复现其对抗子群学习算法，也不把收入当作法律保护身份。

![子群变化](FIGURES/03_subgroup_heatmap.png)

## P2：标签可见性与FRR估计

原反馈台账未发布，因此本轮是静态、独立的新回放，不是旧五轮闭环的复核。仅用各模型Business Group计划审批，拒绝者探索率0/5/10%，五seed共享嵌套uniform。plan无标签输入，先保存后揭示actual获批的成熟结果。IPW估计的拒绝是proposed_reject，探索者不能被误算为原策略批准。估计函数只接受observable ledger，Oracle是最终离线单独函数。

公式：sum_visible I(TARGET0)I(proposed_reject)/pi 除以 sum_visible I(TARGET0)/pi。pi=0的任何目标记录破坏正性，IPW结果为NOT_IDENTIFIABLE；本轮这类组别结果共{int((i.status=='NOT_IDENTIFIABLE').sum())}行。Naive在无探索时通常为0，这不表示真实FRR为0。记录FRR分母相关的TARGET0 ESS以及全部visible ESS、最大权重、覆盖率。0%探索的可见样本即使ESS很高也不具正性，不能把这个ESS误当作完整人群的信息量。归一化比率有限样本可能偏，单个种子也可能IPW更差。

full模型10%探索五seed均值（FRR为比例，ESS为记录数）：

{md(z.round(5).reset_index())}

完整每模型、seed、rate、group和误差95%区间见ipw_estimates.csv。误差bootstrap仅在最终离线评估阶段将oracle与估计配对重采样，oracle不进入估计输入。五seed范围表示同一人群随机探索波动，不能写成五独立现实场景。识别条件包括正确记录倾向、正性、随机探索与成熟机制正确、稳定标签回放及足够ESS。

![IPW误差](FIGURES/04_ipw_estimation_error.png)

## P3：探索风险与信息成本

以下为full模型五seed均值，风险为比例，人数为记录数：

{md(riskfull.round(5).reset_index())}

整个预设回放共有{int(r.actual_exceeds_budget.sum())}/{len(r)}个实际批准风险超过5%，不能把常规5%经验预算写成探索后的系统保证。不同rate均保留，没有用测试结果挑选探索率。若显示ESS增加或误差减少，仍须同时看额外批准和观察风险。Budgeted Exploration为NOT_RUN，本轮不宣称完成共同预算下的探索选择。

![风险与信息](FIGURES/05_exploration_risk_vs_ess.png)

## 新增价值与归因

本项目的贡献是可复核工程组合：同评分同决策空间的双侧保护，有限收入子群的损害检查，以及独立的FRR估计与探索成本审计。没有提出新公平理论，没有复现reductions或RL算法。论文思想与项目实现分别列于REFERENCES.md。P0不改变模型，P1不训练，P2估计可靠性，P3计入探索成本，因果归因不混用。

## 失败、无效与未执行

启动时缺少Path导入导致一次NameError，已修正，尚未训练时发生，日志attempt_01_startup_failure.log保留。macOS缺OpenMP从现有本机库补到项目虚拟环境。sklearn/NumPy在校准中发出浮点matmul RuntimeWarning，未隐藏，需结合calibration_numerics.json的独立元素级梯度检查解释。不可行、未生效、测试违规和负面结果全部保留。旧台账审计BLOCKED_NOT_RUN；可选Budgeted Exploration NOT_RUN；必需P0–P3新实验均已运行。

复现和全部命令见REPRODUCIBILITY_REPORT.md；边界见LIMITATIONS.md。当前为本地CPU验证，没有AWS部署、分布式训练或云端性能测试。
'''
    (HERE/'RESEARCH_EXTENSION_RESULTS.md').write_text(china,encoding='utf-8')
    p=primary_full
    english=f'''# Research Extension V2

## Confirmed Experimental Findings

We recovered six prespecified CPU XGBoost scoring models and evaluated {len(t)} frozen policies on the existing Home Credit historical test split. All {verification['checks']} independent checks passed. The full-model comparison gives test approval rates of {fmt(p.loc[PRIMARY[0],'approval_rate'])}% and {fmt(p.loc[PRIMARY[1],'approval_rate'])}% for Business Group and Fair Gap Only, with absolute FRR gaps of {fmt(p.loc[PRIMARY[0],'abs_frr_gap'])} and {fmt(p.loc[PRIMARY[1],'abs_frr_gap'])} percentage points. The full-model primary Fair Gap + No-Harm policy is INFEASIBLE. Primary no-harm is also infeasible for two of the five biased models. For the other three it selects the same rule as gap-only. No primary model shows an additional feasible no-harm improvement. The main constraints are a 5% validation approved-risk budget, a 2pp FRR-gap cap, and a 1pp maximum groupwise FRR increase relative to the same-model Business Group baseline. These are empirical validation constraints, not future guarantees. Across all presets, {nfeasible} policies were infeasible and {changed} feasible no-harm policies changed the gap-only optimum in sensitivity configurations.

## Exploratory Findings

Seven sensitivity-policy test gaps exceed their frozen validation epsilon, despite validation feasibility. No test-label tuning follows these violations. Six training-tercile income intersections reveal how group averages can conceal local changes. Their paired intervals are conditional and unadjusted for multiple comparisons. Static selective-label replay evaluates 0%, 5%, and 10% exploration over five random seeds. At zero exploration, rejected-label outcomes have zero inclusion probability and group FRR is NOT_IDENTIFIABLE by IPW. With positive propensities, we compare observable-label naive and ratio-IPW estimates against a separate offline oracle, reporting groupwise ESS and estimation error. Actual exploration-inclusive risk exceeds 5% in {int(r.actual_exceeds_budget.sum())} of {len(r)} replay configurations. This count describes repeated simulation on the same records.

## Unresolved Questions

Can a calibrated exploration policy jointly manage information acquisition and total loan risk? Does groupwise validation protection survive temporal drift and finer subgroup definitions? We did not run Budgeted Exploration or a new longitudinal learning experiment. The unavailable original feedback ledgers prevent an independent record-level rerun of the historical 9768 checks.

## Limitations

The test set was previously inspected. Published historical aggregates remain unchanged, but these six restored models are new local checkpoints. Loan IDs do not establish person-level independence. TARGET is a historical repayment-difficulty indicator, not lifetime default or monetary loss. Historical borrowers do not identify real rejected-applicant counterfactuals. Five simulation seeds are not five real lending environments. Group and income definitions are audit strata, not a legal determination of discrimination. This is local CPU validation with no AWS deployment or distributed benchmark. Our contribution is a reproducible engineering combination inspired by established fairness literature, not a new theorem or a full reproduction of reductions, subgroup learning, or long-term reinforcement learning.
'''
    (HERE/'RESEARCH_EXTENSION_RESULTS_EN.md').write_text(english,encoding='utf-8')
    dump(HERE/'presentation/evidence.json',dict(primary_full={k:p.loc[k].to_dict() for k in PRIMARY},biased=biased.to_dict(),
        ipw_full=z.reset_index().to_dict('records'),risk_full=riskfull.reset_index().to_dict('records'),checks=verification['checks'],
        policy_rows=len(t),noharm_changed=changed,infeasible=nfeasible,worst_subgroup=None if worst is None else worst.to_dict(),
        paired_full=stat.to_dict('records'),elapsed_seconds=completion['seconds']))
    artifacts=[p for p in out.rglob('*') if p.is_file() and p.name!='output_hashes.json']
    dump(out/'output_hashes.json',{str(p.relative_to(out)):sha(p) for p in artifacts})
    print('Reports and six PNG/PDF figure pairs generated')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'results/run_001');args=p.parse_args();main(args.run.resolve())
