"""Generate complete tables, paired intervals and publication-exportable figures."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_experiment import HERE,OUT,MODELS,CFG,dump

METRICS=['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk']
MODES={'external':'Independent, matched size','observed':'Observed labels only','frozen':'Frozen initial holdout'}
COLORS={'external':'#168777','observed':'#c45448','frozen':'#6a71a2'}


def paired(pop,left,right):
    arrays=[]
    for l,r in zip(left,right):
        for key in [l,r]:
            with np.load(OUT/'predictions'/f'{key}.npz') as f:arrays.append(f['approved'])
    a=np.column_stack(arrays).astype(np.int64)
    code=(2*pop.thin.to_numpy()+pop.TARGET.to_numpy()).astype(np.int64)
    for j in range(a.shape[1]):code|=a[:,j]<<(j+2)
    unique,counts=np.unique(code,return_counts=True)
    patterns=np.column_stack([(unique>>(j+2))&1 for j in range(a.shape[1])])
    gy=unique&3; thin=(gy==2).astype(float);thick=(gy==0).astype(float);bad=(gy%2).astype(float)
    rng=np.random.default_rng(831)
    weights=rng.multinomial(len(pop),counts/len(pop),size=CFG['bootstrap'])
    def evaluate(w):
        approved=w@patterns
        f1=1-(w@(patterns*thin[:,None]))/(w@thin)[:,None]
        f0=1-(w@(patterns*thick[:,None]))/(w@thick)[:,None]
        risk=np.divide(w@(patterns*bad[:,None]),approved,out=np.full(approved.shape,np.nan),where=approved>0)
        return dict(thin_frr=f1,thick_frr=f0,abs_frr_gap=abs(f1-f0),approval_rate=approved/w.sum(axis=1)[:,None],approved_risk=risk)
    point=evaluate(counts[None,:]);boot=evaluate(weights)
    seeds=rng.integers(0,len(left),size=(len(weights),len(left)))
    rows=[]
    for metric in METRICS:
        delta=boot[metric][:,0::2]-boot[metric][:,1::2]
        draw=np.take_along_axis(delta,seeds,axis=1).mean(axis=1)
        lo,hi=np.nanquantile(draw,[.025,.975])
        rows.append(dict(metric=metric,delta=float((point[metric][:,0::2]-point[metric][:,1::2]).mean()),low=float(lo),high=float(hi)))
    return rows


def intervals(data):
    pop=pd.read_csv(OUT/'test_population.csv');results=[]
    def add(module,setting,comparison,left,right):
        l=left.sort_values('seed').policy_key.tolist();r=right.sort_values('seed').policy_key.tolist()
        assert len(l)==len(r) and len(l) in [1,5]
        for row in paired(pop,l,r):results.append(dict(module=module,setting=setting,comparison=comparison,**row))
    a=data[data.module=='A'];b=data[data.module=='B'];c=data[(data.module=='C')&(data['round']==5)]
    for rate in CFG['retentions']:
        f=a[a.retention==rate]
        for left,right in [('group_only','random'),('biased','group_only'),('corrected','biased')]:
            add('A',str(rate),left+' - '+right,f[f.condition==left],f[f.condition==right])
    for dataset in ['full','biased']:
        f=b[b.dataset==dataset]
        for loss in CFG['losses']:
            if loss!='standard':add('B',dataset,loss+' - standard',f[f.loss==loss],f[f.loss=='standard'])
    for acq in CFG['acquisitions']:
        f=c[c.acquisition==acq]
        for mode in ['external','frozen']:
            add('C',acq,mode+' - observed',f[f['mode']==mode],f[f['mode']=='observed'])
    for mode in CFG['modes']:
        f=c[c['mode']==mode]
        add('C',mode,'explore_ipw - selected',f[f.acquisition=='explore_ipw'],f[f.acquisition=='selected'])
        initial=data[(data.module=='C')&(data['round']==0)&(data['mode']==mode)&(data.acquisition=='selected')]
        add('C',mode,'selected final - initial',f[f.acquisition=='selected'],initial)
    shadow=data[(data.module=='C_shadow')&(data['round']==5)&(data['mode']=='observed')]
    for acq in CFG['acquisitions']:
        f=shadow[shadow.acquisition==acq]
        add('C_shadow',acq,'external - observed',f[f.information=='external'],f[f.information=='observed'])
    result=pd.DataFrame(results);result.to_csv(OUT/'paired_intervals.csv',index=False)
    return result


def table(frame,index):
    keys=['auc']+METRICS
    headers=index+['AUC','薄弱组错拒%','比较组错拒%','差距pp','获批率%','获批风险%']
    rows=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for _,r in frame.iterrows():
        cells=[str(r[k]) for k in index]+[f'{r.auc:.4f}']+[f'{r[k]*100:.2f}' if pd.notna(r[k]) else 'NA' for k in METRICS]
        rows.append('| '+' | '.join(cells)+' |')
    return '\n'.join(rows)


def save(name):
    plt.savefig(HERE/'figures'/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight')
    plt.close()


def plots(a,b,c,shadow,stream,ci):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,rate in enumerate(CFG['retentions']):
        f=a[a.retention==rate].set_index('condition').reindex(['random','group_only','biased','corrected'])
        for k,(metric,title) in enumerate([('thin_frr','Thin-file FRR (%)'),('abs_frr_gap','Absolute FRR gap (pp)')]):
            ax=axes[j,k];ax.bar(range(4),f[metric]*100,color=['#6a8195','#caa452','#c45448','#168777'])
            ax.set_xticks(range(4),['Random N','Group counts','Label distorted','Corrected'],rotation=15,ha='right')
            ax.set_title(f'{title}; retention {rate:.0%}');ax.grid(axis='y',alpha=.2)
    fig.suptitle('A: Separate group representation from within-group label composition');fig.tight_layout();save('01_sampling')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,dataset in enumerate(['full','biased']):
        f=b[b.dataset==dataset].set_index('loss').reindex(CFG['losses'])
        for k,(metric,title) in enumerate([('thin_frr','Thin-file FRR (%)'),('abs_frr_gap','FRR gap (pp)'),('auc','AUC')]):
            ax=axes[j,k];scale=1 if metric=='auc' else 100
            ax.bar(range(5),f[metric]*scale,color=['#6a8195','#c45448','#168777','#7cb7a4','#caa452'])
            ax.set_xticks(range(5),list(CFG['losses']),rotation=25,ha='right',fontsize=8)
            ax.set_title(dataset+': '+title);ax.grid(axis='y',alpha=.2)
            if metric=='auc':ax.set_ylim(.65,.85)
    fig.suptitle('B: Training losses change; approval rule remains the same');fig.tight_layout();save('02_training_losses')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,acq in enumerate(CFG['acquisitions']):
        for k,(metric,title) in enumerate([('thin_frr','Thin-file FRR (%)'),('abs_frr_gap','FRR gap (pp)'),('approval_rate','Approval rate (%)')]):
            ax=axes[j,k]
            for mode in CFG['modes']:
                f=c[(c.acquisition==acq)&(c['mode']==mode)]
                summary=f.groupby('round')[metric].agg(['mean','std'])
                t=summary.index.to_numpy();mean=summary['mean'].to_numpy()*100;sd=summary['std'].to_numpy()*100
                ax.plot(t,mean,'o-',label=MODES[mode],color=COLORS[mode]);ax.fill_between(t,mean-sd,mean+sd,alpha=.12,color=COLORS[mode])
            ax.set(title=acq+': '+title,xlabel='Matured feedback batch');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('C: Validation information sources (bands = SD across five simulations)');fig.tight_layout();save('03_feedback')
    f=shadow[(shadow['round']==5)&(shadow['mode']=='observed')].groupby(['acquisition','information'])[METRICS].mean().reset_index()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,acq in zip(axes,CFG['acquisitions']):
        v=f[f.acquisition==acq].set_index('information').reindex(CFG['modes'])
        ax.bar(range(3),v.thin_frr*100,color=[COLORS[m] for m in CFG['modes']])
        ax.set_xticks(range(3),['Independent','Observed-only','Frozen initial']);ax.set_title(acq+': same final model');ax.set_ylabel('Thin-file FRR (%)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Diagnostic: only calibration / validation information changes');fig.tight_layout();save('04_same_model')
    fig,ax=plt.subplots(figsize=(11,4.8))
    labels=[];values=[]
    for mode in CFG['modes']:
        for acq in CFG['acquisitions']:
            f=stream[(stream['mode']==mode)&(stream.acquisition==acq)]
            labels.append(mode+'\n'+acq);values.append(100*f.bad.sum()/f.approved.sum())
    ax.bar(range(6),values,color=[COLORS[m] for m in CFG['modes'] for _ in range(2)])
    ax.axhline(5,color='black',ls='--',label='5% nominal threshold budget');ax.set_xticks(range(6),labels,fontsize=8)
    ax.set_ylabel('Observed risk among actual approvals (%)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save('05_actual_risk')
    f=ci[(ci.metric=='thin_frr')&(((ci.module=='A')&(ci.setting=='0.5'))|((ci.module=='B')&(ci.setting=='biased')&(ci.comparison=='opportunity1 - standard'))|((ci.module=='C')&(ci.setting=='selected')&(ci.comparison=='external - observed'))|((ci.module=='C_shadow')&(ci.setting=='selected')))]
    fig,ax=plt.subplots(figsize=(12,5))
    labels=[r.module+': '+r.comparison for _,r in f.iterrows()]
    ax.errorbar(f.delta*100,range(len(f)),xerr=np.vstack([(f.delta-f.low)*100,(f.high-f.delta)*100]),fmt='o',capsize=4,color='#168777')
    ax.set_yticks(range(len(f)),labels);ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Thin-file FRR difference (pp), conditional 95% paired interval');ax.grid(axis='x',alpha=.2)
    fig.tight_layout();save('06_primary_effects')


def main():
    data=pd.read_csv(OUT/'metrics.csv');ci=intervals(data)
    composition=[]
    for _,row in data[data.module=='A'].iterrows():
        d=json.loads((MODELS/f'{row.model}_decision.json').read_text(encoding='utf-8'))
        composition.append(dict(model=row.model,retention=row.retention,seed=row.seed,condition=row.condition,
                                total=d['train_size'],thick_good=d['train_cells'][0],thick_bad=d['train_cells'][1],
                                thin_good=d['train_cells'][2],thin_bad=d['train_cells'][3]))
    composition=pd.DataFrame(composition);composition.to_csv(OUT/'sampling_cells.csv',index=False)
    cell_rows=['| 条件 | 比较组0 | 比较组1 | 薄弱组0 | 薄弱组1 | 总量 |','|---|---|---|---|---|---|']
    for _,row in composition[(composition.retention==.5)&(composition.seed==17)].iterrows():
        cell_rows.append('| '+row.condition+' | '+' | '.join(str(int(row[k])) for k in ['thick_good','thick_bad','thin_good','thin_bad','total'])+' |')
    cell_table='\n'.join(cell_rows)
    a=data[data.module=='A'].groupby(['retention','condition'],dropna=False).mean(numeric_only=True).reset_index()
    b=data[data.module=='B'].groupby(['dataset','loss'],dropna=False).mean(numeric_only=True).reset_index()
    c=data[data.module=='C'];s=data[data.module=='C_shadow'];stream=pd.read_csv(OUT/'stream_metrics.csv')
    final=c[c['round']==5].groupby(['mode','acquisition']).mean(numeric_only=True).reset_index()
    for name,frame in [('sampling_summary',a),('loss_summary',b),('feedback_final',final)]:frame.to_csv(OUT/f'{name}.csv',index=False)
    for module in ['A','B','C','C_shadow']:data[data.module==module].to_csv(OUT/f'metrics_{module}.csv',index=False)
    plots(a,b,c,s,stream,ci)
    def effect(module,setting,comparison,metric='thin_frr'):
        r=ci[(ci.module==module)&(ci.setting==setting)&(ci.comparison==comparison)&(ci.metric==metric)].iloc[0]
        return f'{100*r.delta:+.2f}pp，条件95%区间[{100*r.low:+.2f}, {100*r.high:+.2f}]'
    rows=['| 信息模式 | 更新策略 | 实际批准风险% | 探索批准总数/种子 |','|---|---|---|---|']
    for mode in CFG['modes']:
        for acq in CFG['acquisitions']:
            f=stream[(stream['mode']==mode)&(stream.acquisition==acq)]
            rows.append(f'| {mode} | {acq} | {100*f.bad.sum()/f.approved.sum():.2f} | {f.exploration.sum()/5:.1f} |')
    stream_table='\n'.join(rows)
    shadow_final=s[(s['round']==5)&(s['mode']=='observed')].groupby(['acquisition','information']).mean(numeric_only=True).reset_index()
    done=json.loads((OUT/'completion.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8')) if (OUT/'verification.json').exists() else {'checks':'待运行'}
    interpretation=[]
    for mode in CFG['modes']:
        f=c[(c['mode']==mode)&(c.acquisition=='selected')].groupby('round')[METRICS].mean()
        delta=(f.loc[5,'thin_frr']-f.loc[0,'thin_frr'])*100
        direction='升高' if delta>0 else '降低'
        interpretation.append(f'- {mode}：仅获批更新的薄弱组错拒从 {f.loc[0,"thin_frr"]*100:.2f}% 到 {f.loc[5,"thin_frr"]*100:.2f}%，{direction} {abs(delta):.2f}pp；区间见下表。起终点变化不等于逐轮单调变化。')
    c_effects='\n'.join(f'- {mode}，末轮−初始：{effect("C",mode,"selected final - initial")}' for mode in CFG['modes'])
    biased_loss=b[b.dataset=='biased'].set_index('loss')
    full_loss=b[b.dataset=='full'].set_index('loss')
    feedback_end=final.set_index(['mode','acquisition'])
    sampling_main=a[a.retention==.5].set_index('condition')
    text=f'''# 审查补充实验结果：样本构成、验证反馈与训练目标

## 1. 本轮解决了什么

| 审查问题 | 新增设计 | 能回答的范围 |
|---|---|---|
| 1. 人数不足与标签失真混在一起 | 同总量随机、同组别人数且保留组内比例、标签失真、纠正四组 | 在当前受控机制下区分群体构成与组内标签组成 |
| 2. 反馈始终有独立验证支持 | 独立匹配大小、仅可见标签、固定初始三种信息模式；额外同模型诊断 | 比较外部验证支持如何影响反馈轨迹和固定模型的阈值决策 |
| 4. 未改变训练损失 | 标准、业务代价、机会导向及组合加权交叉熵 | 直接比较训练阶段的目标改变；原决策层实验继续作为另一层对照 |

共完成 **{done['models']} 个唯一模型**：A 41个；B新增24个、连同复用基线共30行；C 155个。共701行冻结策略测试评价（含450行同模型信息诊断）。CPU 4线程，训练和评价总耗时 {done['seconds']/60:.1f} 分钟，不含报告、核验。原实验完整保留。

所有新增模型统一200棵树、depth=5；不能将这里的新数值与旧400棵静态实验简单相减归因于本次干预。所有新增训练/阈值先冻结，随后统一评价测试集。该测试集在旧实验已查看，故本轮仍是后续机制研究，不是全新外部盲测。

### 主要观察

1. 主采样场景：group_only薄弱组错拒 {sampling_main.loc['group_only','thin_frr']*100:.2f}%，biased为 {sampling_main.loc['biased','thin_frr']*100:.2f}%。在群体人数相同的情况下，标签组成失真对应更大的错拒变化；人数构成变化本身的区间见第3节。
2. 在biased数据上，机会权重1把差距从 {biased_loss.loc['standard','abs_frr_gap']*100:.2f}pp降至 {biased_loss.loc['opportunity1','abs_frr_gap']*100:.2f}pp，但比较组错拒从 {biased_loss.loc['standard','thick_frr']*100:.2f}%升至 {biased_loss.loc['opportunity1','thick_frr']*100:.2f}%。机会目标改变了分配，不是所有人同时受益。
3. 同一机会权重在full数据上使绝对差距从 {full_loss.loc['standard','abs_frr_gap']*100:.2f}pp变为 {full_loss.loc['opportunity1','abs_frr_gap']*100:.2f}pp。代理目标并不保证缩小实际FRR差距，完整数据上的不利结果不能省略。
4. 仅获批路径末轮，observed的薄弱组错拒 {feedback_end.loc[('observed','selected'),'thin_frr']*100:.2f}%低于external的 {feedback_end.loc[('external','selected'),'thin_frr']*100:.2f}%，但平均测试获批风险分别为 {feedback_end.loc[('observed','selected'),'approved_risk']*100:.2f}%和 {feedback_end.loc[('external','selected'),'approved_risk']*100:.2f}%。验证偏差的后果可能表现为风险约束失准，不能把更低错拒率单独当作更优结论。

## 2. 数据、指标与共同边界

沿用 Home Credit 第三方GitHub镜像和原贷款记录级划分；目标是公开 TARGET 还款困难标签，不等同于终身违约或实际损失金额。薄弱组是0—1条可观察征信记录，比较组至少2条。FRR=标签0中被拒比例，是历史标签下的操作性机会指标；群体差距本身不能证明歧视。

所有审批规则都是在对应验证信息上选择最大批准人数，经验获批风险不超过5%；这不是未来风险保证。新增完整实验不包含真实云服务、真实拒贷反事实、日历时间外验证或真实利润核算。

## 3. A：区分群体构成与组内标签组成

random、group_only、biased 和 corrected 的总人数相同；后三者两组人数也相同。group_only 保持各组原来的标签比例（整数误差不超过1条）；biased 仅删掉薄弱组标签0记录。

以50%保留率、种子17为例，四个标签单元格的训练人数如下。0/1指TARGET，corrected的记录与biased相同，只改变权重：

{cell_table}

{table(a,['retention','condition'])}

主50%场景的薄弱组错拒差值：

- 群体构成效应 group_only−random：**{effect('A','0.5','group_only - random')}**。
- 同人数下标签组成效应 biased−group_only：**{effect('A','0.5','biased - group_only')}**。
- 参考分布纠正 corrected−biased：**{effect('A','0.5','corrected - biased')}**。

现在可以分别观察这两种采样变化的影响，而不能预设人数少就一定造成显著差异。第一项控制总人数、改变群体构成；第二项再固定两组人数，改变组内标签组成。它们不是无条件可加的普遍因果分解。纠正仍依赖代表性有标签参考集、且与人为选择机制匹配，未证明对未知缺失机制普遍有效。

![采样对照](figures/01_sampling.png)

## 4. B：改变训练损失，保持审批规则相同

所有配置使用相同训练记录和模型参数。标准目标是平均二元交叉熵；cost5将标签1的训练损失代价乘5；opportunity1/4分别增加权重1/4的“两组正常标签平均交叉熵”；combined同时采用cost5与机会权重1。公式和等价权重见 PROTOCOL.md，权重进入模型训练梯度，不是训练完再改阈值。

机会项给予两组正常标签相同的组总权重，是机会导向的可加代理目标，**不是直接约束两组FRR相等**。没有保证公平改善、AUC提升或双方同时受益。标准与全部加权模型使用同一独立校准、验证集合及统一风险约束审批规则；原始加权分数需要校准，不能直接当作真实PD。

{table(b,['dataset','loss'])}

五个 biased 数据条件下，相对standard的薄弱组错拒差值：

{chr(10).join('- '+loss+'：'+effect('B','biased',loss+' - standard') for loss in CFG['losses'] if loss!='standard')}

`metrics_B.csv` 同时保留原始Brier、校准Brier、AUC与标签1拒绝率，避免只展示机会指标。完整数据full只有一份基线，不能称作五次独立重复；biased的五个种子反映同一数据内的受控采样变化。代价5和机会权重不是金融机构真实业务参数，所有固定档位均展示，不在测试集上挑最佳档。

**强权重不一定更公平：**biased数据上，机会权重从1增至4后，绝对FRR差距由 {biased_loss.loc['opportunity1','abs_frr_gap']*100:.2f}pp变为 {biased_loss.loc['opportunity4','abs_frr_gap']*100:.2f}pp；两组错拒关系已经反向。AUC同时由 {biased_loss.loc['opportunity1','auc']:.4f}变为 {biased_loss.loc['opportunity4','auc']:.4f}。这说明“优先降低某类错误”的训练代理目标与“差距小”的评价目标并不等价，应把原有决策层硬约束实验一起讲清楚。

![训练目标](figures/02_training_losses.png)

## 5. C：验证信息来源是否改变反馈结论

### 5.1 隔离方式

初始20%训练池有同样的偏置。每条记录在看标签前随机指定80/10/10%的训练/校准/验证角色；后两者永远不入树训练。只有正常批准或随机探索批准后才能获得该条记录标签。所有六条路径共享初始模型、初始校准和验证集，从第一次更新起改变信息来源：

- **external**：独立校准/验证池，人数分别匹配本路径截至当轮可见的保留角色记录数。
- **observed**：仅截至当前可见的保留角色标签，没有独立代表性验证信息。
- **frozen**：固定初始有偏校准/验证记录，作为不补充验证信息的参照。

每种信息模式配 selected 与 explore_ipw 两种数据获取策略。IPW只加权探索路径的树训练记录；保留验证池直接反映可见反馈组成。初始数据不追溯纠正。分支逐轮分化后观察人数也可能不同，因此轨迹比较是完整系统策略的差异。

每批结果假定在下一批前成熟，仍不是现实日历模拟。observed模式也会重复使用历史可见验证标签，可能有适应性过拟合，不能把它当作无偏验证的推荐做法。

### 5.2 第五轮测试结果（五种子均值）

{table(final,['mode','acquisition'])}

{chr(10).join(interpretation)}

{c_effects}

在相同获取策略下，external−observed的薄弱组错拒差值：

- selected：**{effect('C','selected','external - observed')}**。
- explore_ipw：**{effect('C','explore_ipw','external - observed')}**。

**不确定性解释：**本次 selected 路径 external−observed 的末轮薄弱组FRR差值区间跨零，获批风险差值区间也跨零；固定同模型的对应比较亦如此。因此这些均值差异尚不足以支持稳定的方向性结论。当前五个模拟种子仍不足以确定外部验证对这两项指标的平均效应方向，不能把观察均值写成普遍规律。

必须同时检查批准风险和批准人数：如果某路径通过接近全部批准降低FRR，不能据此称其更公平或更优。是否支持“偏差放大”，只按当前信息模式、具体指标和时间范围判断；不为了得到恶化轨迹修改参数。

![反馈轨迹](figures/03_feedback.png)

### 5.3 固定同一个模型，只改变验证信息

以下固定 observed 路径的末轮模型，只替换校准/验证信息来源；不把这些诊断决策回灌到模型。AUC因评分排序相同而相同，审批阈值可以不同。external与observed参照记录数匹配，frozen样本更少。

{table(shadow_final,['acquisition','information'])}

- selected，同模型 external−observed 薄弱组FRR：**{effect('C_shadow','selected','external - observed')}**。
- explore_ipw，同模型 external−observed 薄弱组FRR：**{effect('C_shadow','explore_ipw','external - observed')}**。

这组对照排除了模型评分差异，显示的是验证信息改变决策的效应。正单调校准不改变分数排序；在当前按完整排序前缀选阈值的规则下，关键在验证标签和样本组成，而非校准自动消除了偏见。

![同模型信息诊断](figures/04_same_model.png)

### 5.4 实际放贷风险与探索成本

{stream_table}

风险按五轮、五种子所有实际批准记录汇总，包含探索批准；探索数为每种子五轮平均总数。重复模拟不等于真实新增人口。这里与独立测试的常规策略风险口径不同；5%阈值规则没有约束探索后的整体风险，也不保证biased observed验证能够控制总体风险。

![实际批准风险](figures/05_actual_risk.png)

## 6. 统计、核验与可复现性

所有差值使用同一测试记录配对bootstrap1000次，并配对重采样五个模拟/采样种子；full损失比较只有一个固定模型对。区间条件于当前数据划分、参考和验证机制，不包含重新划分重训、真实人群漂移、自然人聚类或拒贷反事实。多项辅助比较未校正多重检验。全部差值见 `results/paired_intervals.csv`。

![主要差值](figures/06_primary_effects.png)

方法单测检查人数对照、声明损失与权重等价、验证来源。独立核验从台账重建可见数据，检查训练/校准/验证角色隔离、探索概率、IPW、各模式参考来源、阈值可行性与最大批准前缀、测试指标。当前核验检查数：{verification['checks']}，详见 `results/verification.json`。通过核验不等于现实金融系统有效性证明。

## 7. 汇报建议

建议将旧三问结构保留：A新增“同人数、同群体人数”对照；B同时讲训练损失与旧决策目标；C把独立验证支持作为可改变的实验条件，展示所有轨迹及实际批准风险。不要只展示最小错拒率，更不要把机会代理损失叫作已实现严格机会平等。

复现命令、结果目录见 README；实验公式与冻结参数见 PROTOCOL.md。原数据来源和AWS流程对应关系沿用上级README。本地CPU实验并未运行云端审批系统。
'''
    (HERE/'RESULTS_REPORT.md').write_text(text,encoding='utf-8')
    slides=f'''# Presentation notes — audit extensions

## Data: representation vs. label composition

- Match total sample size; then match both group sizes.
- Group-only sampling preserves within-group outcome proportions.
- Thin-file FRR, group-only minus random: {effect('A','0.5','group_only - random')}.
- Thin-file FRR, distorted labels minus group-only: {effect('A','0.5','biased - group_only')}.
- Controlled mechanism evidence; not proof of lender misconduct.

## Training: which errors receive weight?

- Train XGBoost with standard, cost-sensitive, opportunity-weighted and combined log loss.
- Keep records, tree settings, calibration and approval selection rules fixed.
- Opportunity weighting is a surrogate objective, not an equal-FRR constraint.
- Biased data, opportunity1 minus standard FRR: {effect('B','biased','opportunity1 - standard')}.
- Report AUC, both groups, approval rates and risk together.

## Feedback: where do validation labels come from?

- Cross two acquisition policies with three validation information sources.
- Independent matched-size, observed-only, and frozen initial holdouts.
- Holdout roles assigned before observing labels; never train on holdout records.
- Selected-only, independent minus observed-only FRR: {effect('C','selected','external - observed')}.
- Same final observed-only model, changing only validation information: {effect('C_shadow','selected','external - observed')}.
- Outcomes are assumed to mature before each update; actual exploration risk is reported separately.

## Scope

- 220 additional CPU models; all parameters frozen before new evaluation.
- Previously viewed historical test set; conditional paired intervals, no external blind validation.
- Neither low FRR nor small group gaps alone establishes a sound credit policy.
'''
    slides=slides.replace('，条件95%区间','; conditional 95% CI ')
    (HERE/'PRESENTATION_NOTES.md').write_text(slides,encoding='utf-8')
    print('Generated complete report, presentation notes, six PNG/PDF figure pairs and all paired intervals.')


if __name__=='__main__':main()
