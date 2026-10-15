"""Generate paired uncertainty, English figures, and a Chinese evidence-based report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, CFG, dump

COLORS={'full':'#243746','random':'#7896a6','biased':'#c85346','corrected':'#168a7a'}
LABELS={'full':'Full reference','random':'Random removal','biased':'Biased removal','corrected':'Reweighted'}
METRICS=['thin_frr','signed_frr_gap','abs_frr_gap','approval_rate','approved_risk']


def paired_bootstrap(frames, left, right, rate, budget):
    pairs=[]
    for seed in CFG['sampling_seeds']:
        a=frames[f'r{int(rate*100)}_s{seed}_{left}']
        b=frames[f'r{int(rate*100)}_s{seed}_{right}']
        if not np.array_equal(a.SK_ID_CURR,b.SK_ID_CURR): raise ValueError('Pair alignment')
        pairs.extend([a[f'approved_{budget}'].to_numpy(),b[f'approved_{budget}'].to_numpy()])
    # Compress the joint binary outcome pattern across ALL seeds. Resampling these
    # patterns is exactly equivalent to paired customer bootstrap for these metrics.
    approvals=np.column_stack(pairs).astype(np.int64)
    y=a.TARGET.to_numpy(); g=a.thin.to_numpy()
    code=2*g+y
    for j in range(approvals.shape[1]): code=code | (approvals[:,j] << (j+2))
    unique,counts=np.unique(code,return_counts=True)
    gy=(unique & 3)
    patterns=np.column_stack([(unique >> (j+2)) & 1 for j in range(approvals.shape[1])])
    good_thin=(gy==2).astype(float); good_thick=(gy==0).astype(float); bad=(gy%2).astype(float)
    rng=np.random.default_rng(9123+int(rate*100)+int(budget*1000))
    boot=rng.multinomial(len(y),counts/len(y),size=CFG['bootstrap_repetitions'])
    def values(weights):
        approved=weights @ patterns
        thin_frr=1-(weights @ (patterns*good_thin[:,None]))/(weights@good_thin)[:,None]
        thick_frr=1-(weights @ (patterns*good_thick[:,None]))/(weights@good_thick)[:,None]
        risk=np.divide(weights @ (patterns*bad[:,None]),approved,out=np.full_like(approved,np.nan,dtype=float),where=approved>0)
        return {'thin_frr':thin_frr,'signed_frr_gap':thin_frr-thick_frr,'abs_frr_gap':abs(thin_frr-thick_frr),
                'approval_rate':approved/weights.sum(axis=1)[:,None], 'approved_risk':risk}
    original=values(counts[None,:]); simulated=values(boot)
    # Shared seed resampling adds observed training-sampling variability; only five
    # seeds means this is an approximate interval conditional on the fixed split.
    seed_draw=rng.integers(0,len(CFG['sampling_seeds']),size=(len(boot),len(CFG['sampling_seeds'])))
    output=[]
    for metric in METRICS:
        point=(original[metric][:,0::2]-original[metric][:,1::2]).mean()
        delta=simulated[metric][:,0::2]-simulated[metric][:,1::2]
        means=np.take_along_axis(delta,seed_draw,axis=1).mean(axis=1)
        lo,hi=np.nanquantile(means,[.025,.975])
        output.append({'retention':rate,'budget':budget,'comparison':left+' - '+right,'metric':metric,
                       'delta':point,'ci_low':lo,'ci_high':hi})
    return output


def savefig(name):
    plt.savefig(ROOT/f'figures/{name}.png',dpi=180,bbox_inches='tight')
    plt.savefig(ROOT/f'figures/{name}.pdf',bbox_inches='tight')
    plt.close()


def figures(m, ci):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    main=m[np.isclose(m.budget,.05)]
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,metric,title in zip(axs,['thin_frr','abs_frr_gap'],['Thin-file false rejection','Absolute false-rejection gap']):
        baseline=main.loc[main.condition=='full',metric].iloc[0]*100
        ax.axhline(baseline,color=COLORS['full'],ls='--',label=LABELS['full'])
        for condition in ['random','biased','corrected']:
            group=main[main.condition==condition].groupby('retention')[metric].agg(['mean','std']).sort_index()
            ax.errorbar(group.index*100,group['mean']*100,yerr=group['std']*100,marker='o',capsize=4,color=COLORS[condition],label=LABELS[condition])
        ax.set(title=title,xlabel='Thin-file TARGET=0 training retention (%)',ylabel='False rejection (%)' if metric=='thin_frr' else 'Gap (percentage points)')
        ax.grid(alpha=.15)
    axs[0].legend(fontsize=8)
    fig.suptitle('Controlled sampling mechanism | validation risk budget: 5%\nError bars: SD across 5 sampling seeds',fontsize=12)
    fig.tight_layout(); savefig('01_mechanism')
    fig,ax=plt.subplots(figsize=(8,5))
    for condition in LABELS:
        group=main[main.condition==condition]
        ax.scatter(group.approved_risk*100,group.thin_frr*100,s=65 if condition=='full' else 35,color=COLORS[condition],label=LABELS[condition],alpha=.8)
    ax.axvline(5,color='#555555',ls='--',lw=1,label='Validation budget (not test guarantee)')
    ax.set(xlabel='Observed payment difficulties among approved (%)',ylabel='Thin-file false rejection (%)',title='Test outcomes under frozen validation thresholds')
    ax.legend(fontsize=8); ax.grid(alpha=.15); fig.tight_layout(); savefig('02_risk_fairness')
    selected=ci[(ci.retention==.5)&(ci.budget==.05)&(ci.comparison=='corrected - biased')].copy()
    fig,ax=plt.subplots(figsize=(9,4.5))
    positions=np.arange(len(selected))
    ax.errorbar(selected.delta*100,positions,xerr=np.vstack([(selected.delta-selected.ci_low)*100,(selected.ci_high-selected.delta)*100]),fmt='o',capsize=4,color=COLORS['corrected'])
    ax.axvline(0,color='#555555',lw=1)
    ax.set_yticks(positions,['Thin-file FRR','Signed FRR gap','Absolute FRR gap','Approval rate','Approved risk'])
    ax.set(xlabel='Reweighted minus biased (percentage points), 95% interval',title='Main experiment: 50% retention, 5% validation risk budget')
    ax.grid(axis='x',alpha=.15); fig.tight_layout(); savefig('03_correction_effect')
    importance=pd.read_csv(ROOT/'results/feature_importance.csv').head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(importance.feature,importance.mean_abs_shap_logodds,color=COLORS['full'])
    ax.set(xlabel='Mean absolute Tree SHAP contribution (raw log-odds)',title='Full-data model: predictive associations, not causal effects')
    fig.tight_layout(); savefig('04_explanation')
    groups=main[(main.retention==.5)|(main.condition=='full')].groupby('condition')
    names=['full','random','biased','corrected']; positions=np.arange(4)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.bar(positions-.18,[groups.get_group(n).raw_thin_calibration_bias.mean()*100 for n in names],width=.36,label='Before calibration',color='#c85346')
    ax.bar(positions+.18,[groups.get_group(n).thin_calibration_bias.mean()*100 for n in names],width=.36,label='After global calibration',color='#168a7a')
    ax.axhline(0,color='#555555',lw=1); ax.set_xticks(positions,[LABELS[n] for n in names])
    ax.set(ylabel='Mean predicted risk minus observed risk (pp)',title='Thin-file risk estimation | main sampling scenario')
    ax.legend(); fig.tight_layout(); savefig('05_calibration')


def table(rows, columns):
    lines=['| '+' | '.join(label for _,label in columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in rows.iterrows():
        lines.append('| '+' | '.join(str(row[key]) for key,_ in columns)+' |')
    return '\n'.join(lines)


def main():
    done=json.loads((ROOT/'results/completion.json').read_text())
    audit=json.loads((ROOT/'results/data_audit.json').read_text())
    selected=json.loads((ROOT/'results/selection.json').read_text())
    m=pd.read_csv(ROOT/'results/metrics.csv')
    frames={p.name.removesuffix('_test.csv'):pd.read_csv(p) for p in (ROOT/'results/predictions').glob('*_test.csv')}
    outputs=[]
    for rate in CFG['retention_rates']:
        for budget in CFG['risk_budgets']:
            for left,right in [('biased','random'),('corrected','biased')]:
                outputs.extend(paired_bootstrap(frames,left,right,rate,budget))
    ci=pd.DataFrame(outputs); ci.to_csv(ROOT/'results/paired_intervals.csv',index=False)
    summary=m.groupby(['retention','budget','condition']).agg({k:['mean','std'] for k in ['auc','brier','thin_frr','thick_frr','signed_frr_gap','abs_frr_gap','approval_rate','approved_risk']})
    summary.to_csv(ROOT/'results/summary.csv')
    figures(m,ci)
    main_results=m[(m.budget==.05)&((m.retention==.5)|(m.condition=='full'))].groupby('condition').mean(numeric_only=True).reindex(['full','random','biased','corrected']).reset_index()
    main_results['condition']=main_results.condition.map({'full':'完整训练参考','random':'同规模随机删减','biased':'选择性样本偏差','corrected':'联合分布加权纠正'})
    for key in ['thin_frr','thick_frr','signed_frr_gap','abs_frr_gap','approval_rate','approved_risk']:
        main_results[key]=main_results[key].map(lambda v:f'{v*100:.2f}%')
    main_results['auc']=main_results.auc.map(lambda v:f'{v:.4f}')
    main_results['brier']=main_results.brier.map(lambda v:f'{v:.5f}')
    main_results['signed_frr_gap']=main_results.signed_frr_gap.str.replace('%',' pp',regex=False)
    effects=ci[(ci.retention==.5)&(ci.budget==.05)&(ci.comparison=='corrected - biased')]
    frr=effects[effects.metric=='thin_frr'].iloc[0]
    gap=effects[effects.metric=='abs_frr_gap'].iloc[0]
    risk=effects[effects.metric=='approved_risk'].iloc[0]
    statement=('薄弱组错拒率的下降区间未跨过 0，支持该受控机制下存在改善。' if frr.ci_high<0 else
               '薄弱组错拒率的差值区间包含 0，不能断言改善稳定。' if frr.ci_low<=0 else '薄弱组错拒率上升，未实现预期改善。')
    gap_statement=('绝对错拒差距也有稳定下降。' if gap.ci_high<0 else '绝对错拒差距没有得到稳定下降的证据。')
    effects=effects.copy()
    effects['effect']=effects.apply(lambda r:f'{r.delta*100:+.2f} [{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]',axis=1)
    breaches=m[(m.budget==.05)&(m.approved_risk>.05)]
    robust=[]
    for rate in CFG['retention_rates']:
        for budget in CFG['risk_budgets']:
            part=m[(m.retention==rate)&(m.budget==budget)]
            before=part[part.condition=='biased'].mean(numeric_only=True)
            after=part[part.condition=='corrected'].mean(numeric_only=True)
            robust.append({'retention':f'{rate:.0%}','budget':f'{budget:.0%}',
                           'before':f'{before.thin_frr:.2%}','after':f'{after.thin_frr:.2%}',
                           'risk_before':f'{before.approved_risk:.2%}','risk_after':f'{after.approved_risk:.2%}'})
    robust=pd.DataFrame(robust)
    main_numeric=m[(m.retention==.5)&(m.budget==.05)].groupby('condition').mean(numeric_only=True)
    thick_change=main_numeric.loc['corrected','thick_frr']-main_numeric.loc['biased','thick_frr']
    population=pd.read_csv(ROOT/'data/processed/split_manifest.csv').groupby('thin').TARGET.agg(['size','mean'])
    txt=f'''# 实验结果报告：信用记录薄弱客户的样本偏差与纠正

## 1. 结论

本机 CPU 已完成 {done['models']} 个正式对照模型，另有 4 个超参数候选模型。主实验保留薄弱记录组 TARGET=0 样本的 50%，以验证集获批者还款困难率不超过 5% 选择统一审批阈值。

- 纠正相对偏置模型：薄弱组错拒率变化 **{frr.delta*100:+.2f} 个百分点**，近似 95% 配对区间 **[{frr.ci_low*100:+.2f}, {frr.ci_high*100:+.2f}]**。
- {statement}{gap_statement}
- 获批者还款困难率变化 **{risk.delta*100:+.2f} 个百分点**，区间 **[{risk.ci_low*100:+.2f}, {risk.ci_high*100:+.2f}]**；应结合下面的风险与机会结果解释。
- 比较组错拒率变化 **{thick_change*100:+.2f} 个百分点**。修复减少了薄弱组的额外错拒，同时也改变了比较组的批准机会，并非所有客户都受益。
- 在 5% 验证预算下，31 个模型中 **{len(breaches)} 个**在测试集的观察风险超过 5%。验证约束不构成未来风险保证。

这是受控样本选择机制的实验结论，不是对 Home Credit 历史歧视行为的认定。

## 2. 官方参考、真实数据与本地实现

流程参考 AWS 官方信用风险预测、解释和偏差缓解项目。AWS 使用 South German Credit、XGBoost 和 SageMaker；本实验使用 Home Credit 数据与本机 XGBoost，采用独立参考集估计联合分布权重。没有运行 AWS 云服务，也没有复现其多目标调参算法；不能称为 AWS 或银行的生产模型。

数据通过第三方 GitHub 镜像获取，申请表 {audit['applications']:,} 行、原始 122 列；征信表 {audit['bureau_rows']:,} 行、17 列。原始阳性标签 {audit['target_1']:,} 个，薄弱组 {audit['thin_count']:,} 人。镜像代码显示 CSV 直接转换成 Parquet；结构及标签核验通过，但没有官方字节哈希可验证逐值完全一致。文件网址、固定提交和 SHA-256 见 `data/raw/source.json`。

原始数据审计：薄弱组的观察还款困难率为 {population.loc[1,'mean']:.2%}，比较组为 {population.loc[0,'mean']:.2%}。因此不能把所有原始审批差距都解释为歧视；完整数据基线也保留了组间差异。

TARGET=1 指在前 Y 期还款中至少一期逾期超过 X 天；X、Y 未公开。TARGET=0 是其余情况，下文“正常/错拒”均相对此标签，不代表永久不会违约，也不是实际经济损失。

## 3. 实验设计与防止信息泄漏

- 研究组：申请前可观察征信记录 0—1 条；比较组：至少 2 条。零记录可能源于数据覆盖，不能等同于现实中无信用历史。该分组并非法律上的受保护属性。
- 按贷款记录分成训练 50%、参考 10%、概率校准 10%、验证 10%、测试 20%，按组别×标签分层，ID 无交叉。
- 输入包含申请信息及征信聚合，排除 ID、TARGET 和性别。保留缺失信息；处理工作天数哨兵值，增加收入负担比。征信未来开立/更新记录排除 {audit['bureau_future_rows_excluded']} 条；没有使用未来合同到期日推断实际未来还款。数据没有绝对申请时间，无法做跨年份验证。
- 特征编码仅在完整训练集拟合，所有条件共享词表与特征空间。XGBoost 原生处理数值缺失。超参数只用完整训练集和验证集选择，锁定为 `{selected['parameters']}`。
- 选择性删减只针对薄弱组 TARGET=0，保留率 50% / 25%，每档 5 个固定种子。真实标签不改；同规模随机对照具有相同训练人数。
- 纠正模型与偏置模型使用完全相同记录。权重为参考集 P(组别,标签) / 偏置训练集 P(组别,标签)，不强行让组别与风险标签独立，不使用测试标签或已知保留率估计权重。
- 所有模型在独立校准集上做全局 Platt 校准，再在验证集选批准人数最多且观察风险不超过预算的统一阈值，至少需 500 个验证获批者；没有可行阈值就拒绝全部，不偷偷放宽预算。相同分数整体接受或拒绝。
- 所有模型与阈值先保存到 `frozen_decisions.json`，之后才计算测试预测与指标。3%、8% 是预先规定的补充预算。

## 4. 主实验测试结果

随机、偏置、纠正行是 5 个采样种子的均值；完整训练参考只有一个模型。FRR 为 TARGET=0 中被拒绝比例。正的有符号差距意味着薄弱组错拒更多，绝对差距另行衡量。

{table(main_results,[('condition','条件'),('auc','AUC'),('approval_rate','获批率'),('approved_risk','获批风险'),('thin_frr','薄弱组错拒'),('thick_frr','比较组错拒'),('signed_frr_gap','有符号差距'),('brier','Brier')])}

### 纠正减去偏置：主实验差值

{table(effects,[('metric','指标'),('effect','变化与 95% 区间（百分点）')])}

区间使用 {CFG['bootstrap_repetitions']} 次配对重采样：同一测试贷款记录在所有模型中同步重采样，并重采样 5 个训练采样种子。没有重新分割数据或重新训练每个 bootstrap 模型，故区间条件于当前划分、参考集和校准集；仅 5 个种子，训练不确定性估计有限。多个辅助指标没有多重检验校正。

完整的 25% 压力实验、3% / 8% 预算以及“偏置−同规模随机对照”的差值均保留在 `results/metrics.csv`、`summary.csv`、`paired_intervals.csv`。不得只挑最显著的场景展示。

### 所有预设场景的纠正前后均值

{table(robust,[('retention','样本保留率'),('budget','验证风险预算'),('before','偏置薄弱组错拒'),('after','纠正薄弱组错拒'),('risk_before','偏置获批风险'),('risk_after','纠正获批风险')])}

8% 预算接近样本总体风险，模型几乎批准全部客户，所以纠正的错拒改善幅度明显缩小；部分模型的测试风险略超过 8%。这说明公平性效果依赖审批政策，也再次说明验证预算不能保证测试风险达标。

## 5. 图表与解释

![样本偏差与纠正](figures/01_mechanism.png)

![风险与机会](figures/02_risk_fairness.png)

![纠正差值区间](figures/03_correction_effect.png)

![特征关联解释](figures/04_explanation.png)

![概率校准](figures/05_calibration.png)

Tree SHAP 来自完整训练参考模型，使用预设随机的 1,500 个测试特征行；没有用于选模。贡献单位为未校准 log-odds，显示预测关联，不能证明某字段导致歧视。全局校准是所有条件共同的操作；正单调校准不改变排序，在验证排序阈值策略下不会自行改变可选批准集合，因此不能把其与样本纠正混为同一个干预。

## 6. 局限与业务含义

1. 原始数据并非全体申请者的随机样本，也不含被拒客户的可观察还款反事实；完整训练模型只是当前数据内的参考，并非无偏真值。
2. 受控删减展示的是一个设定机制。它没有证明金融机构实际按此方式采样，也不等价于拒绝推断已被解决。
3. 权重纠正依赖代表性且有标签的参考样本；真实业务中可能需要独立审计或长期跟踪，不能凭空获得。
4. 更低组间差距并不自动意味着更正确的决策，必须同时看薄弱组错拒、总体获批、风险和概率校准。群体真实风险差异可以保留。
5. 风险预算由验证集经验值确定，可能发生过拟合或漂移；5% 为课堂设定，非 Home Credit 标准。没有损失率、利率和资金成本，不能报告真实利润收益。
6. 本地 CPU 展示的是云端 ML 流程的本地实现，不涉及云部署、分布式性能或在线监控的实测。汇报可把这些列作部署扩展。

## 7. 复现与来源

- Python / 包版本见 `requirements-lock.txt` 与 `results/completion.json`；设备 CPU，线程 {CFG['threads']}。
- 31 个正式模型训练、校准及验证累计 {done['sum_training_seconds']:.1f} 秒；本次完整主程序耗时 {done['wall_seconds_this_invocation']:.1f} 秒（不含下载、依赖安装和报告 bootstrap）。
- 执行 `./run.ps1` 完成数据检查、方法测试、训练、图表报告与独立指标核验；重复执行可复用同一配置和代码的模型检查点。
- 独立核验结果见 `results/verification.json`。README 说明数据下载与目录结构。
- [AWS 官方参考](https://github.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection)
- [Home Credit 原始发布](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- [GitHub 数据镜像](https://github.com/qwyt/HomeCreditDefaultRisk/tree/920bbfca12ac236a10ca6b9772286fa15f7a9cca/dataset/full)
- [镜像格式转换源码](https://github.com/qwyt/HomeCreditDefaultRisk/blob/920bbfca12ac236a10ca6b9772286fa15f7a9cca/src_draft/load_dataset.py)
'''
    (ROOT/'RESULTS_REPORT.md').write_text(txt,encoding='utf-8')
    print('Generated paired intervals, 5 PNG/PDF figure pairs, and RESULTS_REPORT.md.')


if __name__=='__main__': main()
