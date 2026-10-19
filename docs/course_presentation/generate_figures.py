"""Re-express saved experimental results for the course README; no fitting/selection."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/course_presentation'
TEAL, RUST, SLATE, GOLD = '#167c83', '#bd5b43', '#536c83', '#b28a36'
INPUTS = {
    'sampling': ROOT/'audit_extension/results/sampling_summary.csv',
    'loss': ROOT/'audit_extension/results/loss_summary.csv',
    'policy': ROOT/'research_extension_v2/results/run_001/COMPARISON_TABLES.csv',
    'interval': ROOT/'research_extension_v2/results/run_001/paired_intervals.csv',
    'subgroup': ROOT/'research_extension_v2/results/run_001/subgroups.csv',
    'ipw': ROOT/'research_extension_v2/results/run_001/ipw_estimates.csv',
    'risk': ROOT/'research_extension_v2/results/run_001/exploration_risk.csv',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(fig, stem, note):
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=10, color='#425361')
    fig.tight_layout(rect=(0, .07, 1, .91))
    fig.savefig(OUT/f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def labels(ax, bars, suffix=''):
    ax.bar_label(bars, labels=[f'{v:.2f}{suffix}' for v in bars.datavalues], padding=6, fontsize=12)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {k: sha(p) for k,p in INPUTS.items()}
    tables = {k: pd.read_csv(p) for k,p in INPUTS.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.axisbelow':True,
        'axes.titlesize':14, 'axes.titleweight':'bold', 'axes.labelcolor':'#253c49',
        'text.color':'#253c49', 'xtick.color':'#253c49', 'ytick.color':'#253c49'})
    data = {}
    # 1. Historical sampling and loss interventions use their own matched controls.
    sampling = tables['sampling'].query('retention == 0.5').set_index('condition')
    loss = tables['loss'].query("dataset == 'biased'").set_index('loss')
    fig, axes = plt.subplots(1,2,figsize=(12,5.2))
    order=['random','group_only','biased','corrected']
    b=axes[0].bar(range(4),sampling.loc[order,'thin_frr']*100,color=[SLATE,GOLD,RUST,TEAL],width=.66)
    labels(axes[0],b);axes[0].set_xticks(range(4),['Random\nsame N','Matched\ngroup N','Distorted\nlabels','Reference\nreweighting'])
    axes[0].set(title='Sampling mechanism',ylabel='Thin-group test FRR (%)',ylim=(0,48));axes[0].grid(axis='y',alpha=.2)
    for j,(column,name,color) in enumerate([('thin_frr','Thin',TEAL),('thick_frr','Comparison',SLATE)]):
        values=loss.loc[['standard','opportunity1'],column]*100
        b=axes[1].bar(np.arange(2)+(j-.5)*.32,values,width=.32,label=name,color=color);labels(axes[1],b)
    axes[1].set_xticks([0,1],['Standard loss','Opportunity weight 1'])
    axes[1].set(title='Changing the training loss',ylabel='Test FRR (%)',ylim=(0,48));axes[1].legend(loc='upper right');axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Historical experiments: label composition and training objectives',fontsize=16,fontweight='bold')
    finish(fig,'01_sampling_and_training','audit_extension | 50% thin normal-label retention | Five sampling-seed means on one fixed test split')
    data['historical']={'sampling':sampling.loc[order,['thin_frr','thick_frr','abs_frr_gap']].to_dict('index'),
        'loss':loss.loc[['standard','opportunity1','opportunity4'],['thin_frr','thick_frr','abs_frr_gap']].to_dict('index')}
    # 2. The V2 fixed-score policy comparison is distinct from loss interventions.
    policy=tables['policy'];interval=tables['interval']
    full=policy[policy.model=='full'].set_index('policy');fig,ax=plt.subplots(figsize=(9,5.2))
    keys=['business','gap_e0.02']
    for j,(key,name,color) in enumerate(zip(keys,['Business Group','Fair Gap Only'],[SLATE,TEAL])):
        values=np.array([full.loc[key,'thin_frr'],full.loc[key,'thick_frr']])*100
        rows=interval[(interval.model=='full')&(interval.policy==key)].set_index('metric')
        low=rows.loc[['thin_frr','thick_frr'],'lo'].to_numpy()*100
        high=rows.loc[['thin_frr','thick_frr'],'hi'].to_numpy()*100
        b=ax.bar(np.arange(2)+(j-.5)*.3,values,width=.3,color=color,label=name,
            yerr=np.stack([values-low,high-values]),capsize=4,error_kw={'linewidth':1})
        labels(ax,b)
    ax.set_xticks([0,1],['Thin group','Comparison group']);ax.set(ylabel='Test FRR (%)',ylim=(0,25))
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right')
    fig.suptitle('A smaller group gap can increase rejection in the comparison group',fontsize=15,fontweight='bold')
    finish(fig,'02_fixed_score_policies','V2 full fixed model | Bars: point estimates; error bars: conditional paired-bootstrap 95% intervals\nPrimary gap + no-harm policy: INFEASIBLE on the validation grid')
    data['full_policy']=full.loc[keys,['thin_frr','thick_frr','abs_frr_gap','approval_rate','approved_risk','auc']].to_dict('index')
    # 3. Show every primary model, including infeasible no-harm decisions.
    models=['full']+[f'r50_s{s}_biased' for s in [17,29,43,61,89]]
    primary=['business','gap_e0.02','noharm_e0.02_d0.01']
    vals=[];text=[]
    for model in models:
        r=policy[policy.model==model].set_index('policy');v=[];tt=[]
        for key in primary:
            passed=r.loc[key,'status']=='PASS';v.append(int(passed))
            tt.append('FEASIBLE' if key!=primary[-1] else ('SAME RULE' if passed else 'INFEASIBLE'))
        vals.append(v);text.append(tt)
    fig,ax=plt.subplots(figsize=(9.3,5.1))
    ax.imshow(vals,cmap=ListedColormap(['#f3d8c8','#dcece8']),vmin=0,vmax=1,aspect='auto')
    for i in range(6):
        for j in range(3):ax.text(j,i,text[i][j],ha='center',va='center',fontsize=12,fontweight='bold',color='#273f48')
    ax.set_xticks(range(3),['Business Group','Fair Gap Only','Gap + No-Harm'])
    ax.set_yticks(range(6),['Full model']+[f'Biased model / seed {s}' for s in [17,29,43,61,89]])
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    fig.suptitle('Primary no-harm: three infeasible models, three unchanged rules',fontsize=15,fontweight='bold')
    finish(fig,'03_noharm_feasibility','Validation constraints: risk <= 5%, gap <= 2pp, each group deterioration <= 1pp\nSAME RULE = feasible, but identical to Fair Gap Only; infeasibility refers to the frozen finite grid')
    data['primary_status']=[dict(model=m,**dict(zip(primary,t))) for m,t in zip(models,text)]
    # 4. A six-cell view of the prespecified full-model income intersections.
    sub=tables['subgroup'].query("model == 'full' and policy == 'gap_e0.02'")
    groups=['thin','comparison'];bands=['low','middle','high'];changes=np.zeros((2,3))
    fig,ax=plt.subplots(figsize=(10,4.9));records=[]
    for a,group in enumerate(groups):
        for b,band in enumerate(bands):
            row=sub[(sub.group==group)&(sub.income==band)].iloc[0];changes[a,b]=row.frr_change*100
            r=interval[(interval.model=='full')&(interval.policy=='gap_e0.02')&
                (interval.metric==f'subgroup_{1 if group=="thin" else 0}_{b}')].iloc[0]
            records.append(dict(group=group,income=band,change_pp=row.frr_change*100,
                lo_pp=r.difference_lo*100,hi_pp=r.difference_hi*100,normal_count=int(row.normal_count)))
    im=ax.imshow(changes,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-5,vcenter=0,vmax=5),aspect='auto')
    for a in range(2):
        for b in range(3):
            r=records[a*3+b];color='white' if abs(r['change_pp'])>3 else '#172e39'
            ax.text(b,a,f"{r['change_pp']:+.2f} pp\n[{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]",ha='center',va='center',fontsize=13,color=color)
    ax.set_xticks(range(3),['Low income','Middle income','High income']);ax.set_yticks([0,1],['Thin group','Comparison group'])
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    fig.colorbar(im,ax=ax,pad=.03,label='FRR change (pp)')
    fig.suptitle('Group averages can conceal income-subgroup deterioration',fontsize=15,fontweight='bold')
    finish(fig,'04_subgroup_changes','Full fixed model | Gap Only minus same-model Business Group | Income boundaries use training data only\nBrackets: conditional paired-bootstrap 95% intervals; exploratory and unadjusted for multiple comparisons')
    data['subgroup']=records
    # 5. Seed ranges are Monte Carlo ranges, not 95% confidence intervals.
    ipw=tables['ipw'].query("model == 'full' and exploration_rate == 0.1")
    fig,ax=plt.subplots(figsize=(10,5.4));stats=[]
    for j,(method,name,color) in enumerate([('naive_frr','Naive observed labels',GOLD),('ipw_frr','IPW',TEAL),('oracle_frr','Offline oracle',SLATE)]):
        means=[];mins=[];maxs=[]
        for group in [1,0]:
            q=ipw[ipw.group==group][method]*100
            means.append(q.mean());mins.append(q.min());maxs.append(q.max())
            stats.append(dict(group=group,method=method,mean_pct=q.mean(),min_pct=q.min(),max_pct=q.max()))
        means=np.array(means);b=ax.bar(np.arange(2)+(j-1)*.23,means,width=.23,color=color,label=name,
            yerr=np.stack([means-mins,maxs-means]),capsize=4,error_kw={'linewidth':1});labels(ax,b)
    ax.set_xticks([0,1],['Thin group','Comparison group']);ax.set(ylabel='Estimated original-policy FRR (%)',ylim=(0,26));ax.grid(axis='y',alpha=.2)
    ax.legend(loc='upper right',fontsize=11)
    fig.suptitle('Observed labels can severely understate original-policy rejection',fontsize=15,fontweight='bold')
    finish(fig,'05_selective_label_estimates','Full fixed model | Explore 10% of planned rejections | Five-seed means; whiskers show seed min-max, not 95% CI\nOracle is an offline historical replay benchmark; 0% exploration leaves IPW NOT_IDENTIFIABLE')
    data['ipw_10pct']=stats
    # 6. Rates have different denominators: actual total versus exploration subset.
    risk=tables['risk'].query("model == 'full'");rg=risk.groupby('exploration_rate')
    summary=rg[['regular_n','regular_risk','exploration_n','exploration_risk','actual_n','actual_risk']].mean()
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.2))
    q=rg.actual_risk.agg(['mean','min','max'])*100
    axes[0].errorbar(q.index*100,q['mean'],yerr=np.stack([q['mean']-q['min'],q['max']-q['mean']]),
        color=TEAL,marker='o',capsize=4,linewidth=2)
    for x,y in zip(q.index*100,q['mean']):axes[0].annotate(f'{y:.2f}%',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=12)
    axes[0].axhline(5,color=RUST,linestyle='--',label='5% validation budget')
    axes[0].set(title='Total actual approvals',xlabel='Explore planned rejections (%)',ylabel='Observed historical risk (%)',ylim=(4.65,5.4))
    axes[0].set_xticks([0,5,10]);axes[0].grid(axis='y',alpha=.2);axes[0].legend(loc='lower right',fontsize=10)
    r10=summary.loc[.1];b=axes[1].bar([0,1],[r10.regular_risk*100,r10.exploration_risk*100],color=[SLATE,RUST],width=.55);labels(axes[1],b,'%')
    axes[1].set_xticks([0,1],['Regular\napprovals','Exploration-only\napprovals']);axes[1].set(title='Approval subsets at 10% exploration',ylabel='Observed historical risk (%)',ylim=(0,27));axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('More feedback can come with additional lending risk',fontsize=16,fontweight='bold')
    finish(fig,'06_exploration_risk','Full fixed model | Five-seed means; left whiskers show seed min-max, not 95% CI\nRisk = TARGET=1 share within the named approval set; validation budget does not guarantee future or total actual risk')
    data['risk_full']=summary.reset_index().replace({np.nan:None}).to_dict('records')
    after={k:sha(p) for k,p in INPUTS.items()}
    assert before==after,'Saved experiment input changed'
    manifest={'purpose':'Course README figures derived from frozen aggregates; no new training or policy selection',
        'inputs':{str(p.relative_to(ROOT)):before[k] for k,p in INPUTS.items()},'generator_sha256':sha(Path(__file__)),
        'data':data,'outputs':{str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('*')) if p.suffix in ['.png','.pdf']}}
    (OUT/'figure_data.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('PASS: six PNG/PDF pairs generated from unchanged saved tables')


if __name__=='__main__':main()
