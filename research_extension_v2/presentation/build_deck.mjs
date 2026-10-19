import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const here=path.dirname(fileURLToPath(import.meta.url));
const skill=process.env.PRESENTATION_SKILL_DIR ?? '/Users/pytenter/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
const runtimePython=process.env.RUNTIME_PYTHON ?? '/Users/pytenter/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const font=resolvePresentationFont();
const evidence=JSON.parse(await fs.readFile(path.join(here,'evidence.json'),'utf8'));
const content=JSON.parse(await fs.readFile(path.join(here,'content.json'),'utf8'));
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#172b3a',muted='#526270',teal='#167d86',orange='#b86b2d',bg='#faf9f6';
const pct=x=>(x*100).toFixed(2);
function text(slide,value,x,y,w,h,size=28,color=ink,bold=false){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=[[{run:value,style:{typeface:font,fontSize:`${size}px`,color,bold}}]];s.text.style={fontSize:size,color,bold};return s;
}
function add(index,kicker){
 const s=presentation.slides.add();s.background.fill=bg;
 text(s,content[index].title,64,42,1150,78,40,ink,true);
 if(kicker)text(s,kicker,64,120,1150,36,21,teal,true);
 text(s,`${index+1} / 8`,1160,665,70,28,17,muted);
 s.speakerNotes.text=content[index].speech+'\n\nSources (not spoken):\n'+content[index].sources;
 return s;
}
function table(s,values,x,y,w,h,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.value=[[{run:String(values[r][c]),style:{typeface:font,fontSize:'23px',color:r===0?'#ffffff':ink,bold:r===0}}]];cell.text.style={fontSize:23,color:r===0?'#ffffff':ink,bold:r===0};cell.fill=r===0?ink:(r%2?'#f1f1ed':bg);
 }
 t.borders.assign({fill:'#d6dad8',width:.6,style:'solid'});return t;
}
function chart(s,categories,series,x,y,w,h,yTitle,max){
 series=series.map(z=>({...z,values:z.values.map(v=>Number(v.toFixed(2)))}));
 const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series,
  hasLegend:series.length>1,legend:{position:'bottom',textStyle:{fontSize:20}},
  barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
  chartFill:bg,plotAreaFill:bg,
  xAxis:{textStyle:{fontSize:20,fill:ink}},
  yAxis:{min:0,max:max,title:{text:yTitle,textStyle:{fontSize:20}},numberFormatCode:'0.0',textStyle:{fontSize:18,fill:muted},majorGridlines:{fill:'#dde1df',width:.5}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:20,fill:ink}},
 });applyPresentationChartFont(c,{fontFamily:font});return c;
}
// 1: minimal cover.
{
const s=presentation.slides.add();s.background.fill=ink;
text(s,'Algorithmic Discrimination\nin Credit Approval',76,154,1125,220,66,'#ffffff',true);
text(s,'Fairness interventions, costs, and reliable evaluation',80,410,1090,70,32,'#9ed4d3');
text(s,'Cloud Computing Course Project\nHome Credit historical replay. Local CPU validation',80,575,1100,72,22,'#d4dfe4');
s.speakerNotes.text=content[0].speech+'\n\nSources (not spoken):\n'+content[0].sources;
}
// 2: background and the requested lifecycle.
{
const s=add(1,'Part 1: Background and Current State of Research');
text(s,'Opportunity depends on the decision rule',64,185,625,65,32,ink,true);
text(s,'FRR = P(reject | TARGET = 0, group)\n\nThin: 0–1 observable bureau records\nComparison: at least 2 records\n\nTARGET = 1: historical repayment difficulties\nTARGET = 0: historical normal label',64,265,665,310,27);
text(s,'Data collection\n\nData processing\n\nModel training\n\nFairness-aware decision\n\nFeedback monitoring',815,181,390,415,28,teal,true);
text(s,'A small group gap can hide harm. Observed labels can hide bias.',64,624,1100,35,23,orange);
}
// 3: seven formal sources with their contributions.
{
const s=add(2);
text(s,'Decisions and constraints',64,166,555,42,29,teal,true);
text(s,'Hardt et al. (2016)\nOpportunity and post-processing\n\nCorbett-Davies et al. (2017)\nUtility under fairness constraints\n\nAgarwal et al. (2018)\nExplicit constrained classification',64,224,555,380,26);
text(s,'Subgroups and selective labels',690,166,530,42,29,teal,true);
text(s,'Kearns et al. (2018): hidden subgroups\n\nLakkaraju et al. (2017): selective labels\n\nKilbertus et al. (2020): decision policies\n\nValdrighi et al. (2026): long-term fairness',690,224,530,330,25);
text(s,'Project scope: a reproducible engineering combination inspired by these ideas',64,620,1120,36,23,muted);
}
// 4: original framework, controlled recovery.
{
const s=add(3);
text(s,'307,511 labeled applications and 1,716,428 bureau records',64,156,1150,50,31,teal,true);
table(s,[['Training','Reference','Calibration','Validation','Test'],['50%','10%','10%','10%','20%']],64,237,1150,135,[230,230,230,230,230]);
text(s,'Existing study',64,416,500,45,30,ink,true);
text(s,'220 model identifiers, 701 published evaluations\nSampling, training objectives, and feedback',64,477,550,105,26);
text(s,'This follow-up',695,416,500,45,30,ink,true);
text(s,'6 recovered fixed scoring models, 78 policies\n200 trees, depth 5, CPU with 4 threads',695,477,520,105,26);
text(s,'All new rules freeze before evaluation. The historical test set was previously inspected.',64,624,1135,38,22,orange);
}
// 5: native editable FRR chart and measured costs.
{
const s=add(4,'Part 2: Personal Insights and Feasible Solutions');
const b=evidence.primary_full.business,f=evidence.primary_full['gap_e0.02'];
chart(s,['Business Group','Fair Gap Only'],[
{name:'Thin',values:[b.thin_frr*100,f.thin_frr*100],fill:teal},
{name:'Comparison',values:[b.thick_frr*100,f.thick_frr*100],fill:'#62778a'}],64,182,730,400,'Test FRR (%)',25);
text(s,`${pct(b.abs_frr_gap)} to ${pct(f.abs_frr_gap)} pp`,845,205,360,70,38,teal,true);
text(s,'Absolute FRR gap',845,286,340,42,25);
text(s,`+${pct(f.thick_frr-b.thick_frr)} pp`,845,375,340,58,38,orange,true);
text(s,'Comparison-group FRR change',845,446,345,78,25);
text(s,`Full fixed model. Approval: ${pct(b.approval_rate)}% to ${pct(f.approval_rate)}%. Approved risk: about 4.79%. Same AUC: 0.7674.`,64,615,1120,55,22,muted);
}
// 6: editable constraints and explicit negative finding.
{
const s=add(5);
text(s,'Maximize approval under shared constraints',64,160,1140,48,31,teal,true);
table(s,[['Constraint','Frozen validation requirement'],['Approved risk','At most 5%'],['Absolute FRR gap','At most 2 percentage points'],['Each group’s FRR','At most Business Group + 1pp']],64,241,750,270,[240,510]);
text(s,'3 of 6',876,240,320,80,58,orange,true);
text(s,'INFEASIBLE',876,337,325,45,28,orange,true);
text(s,'Other 3: same rule as gap-only',876,417,310,100,25);
text(s,'Same model, same candidate grid, same-model Business Group baseline. No constraint relaxation.',64,586,1120,70,24,muted);
}
// 7: separate estimation and risk outcomes.
{
const s=add(6);
text(s,'Static replay on one historical population. Five simulation seeds.',64,138,1140,42,24,muted);
text(s,'Thin-group FRR at 10% exploration',64,206,550,42,28,teal,true);
const ipw=evidence.ipw_full.find(x=>x.group===1);
chart(s,['Naive','IPW','Offline oracle'],[{name:'FRR',values:[ipw.naive_frr*100,ipw.ipw_frr*100,ipw.oracle_frr*100],fill:teal}],64,270,540,290,'FRR (%)',25);
text(s,'Actual approved risk includes exploration',690,206,530,42,27,orange,true);
chart(s,['0%','5%','10%'],[{name:'Risk',values:evidence.risk_full.map(x=>x.actual_risk*100),fill:orange}],690,270,520,290,'Approved risk (%)',6);
text(s,'0% exploration: IPW NOT_IDENTIFIABLE\nIncome audit: comparison / middle FRR rises 1.82pp (exploratory)',64,602,1135,64,23,muted);
}
// 8: caveats in the conclusion itself.
{
const s=add(7);
text(s,'Fairness decisions have distributional costs',64,166,1120,65,36,teal,true);
text(s,'Groupwise protection can be infeasible\nExploration trades information for lending risk\nMonitoring needs visible-label probabilities and ESS',64,266,1130,185,30);
text(s,'Evidence boundary',64,496,1100,42,28,ink,true);
text(s,'Previously inspected historical test data. No real rejected-applicant counterfactuals.\nLocal CPU validation. Cloud architecture remains a proposed extension.\nNo AWS deployment or distributed performance benchmark.',64,554,1120,99,24,muted);
}
await fs.mkdir(path.join(here,'.build'),{recursive:true});await fs.mkdir(path.join(here,'output'),{recursive:true});
for(let i=0;i<presentation.slides.items.length;i++){
 const slide=presentation.slides.items[i];
 const blob=await presentation.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(here,'.build',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidatePath=path.join(here,'.build','candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const result=await finalizePresentation({workspaceDir:here,candidatePath,finalPath:path.join(here,'output','Fairness_Research_V2.pptx'),
 pythonExecutable:runtimePython,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','6'],
 explicitTotalSlideCount:8,requiredNativeTableOwnerSlides:[4,6],requiredNativeChartOwnerSlides:[5,7],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
 receiptPath:path.join(here,'.build','validation.json')});
console.log(JSON.stringify({font,finalPath:path.join(here,'output','Fairness_Research_V2.pptx'),result}));
