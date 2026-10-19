# 可复现性报告

## 身份、环境、输入与保留

- 日期：2026-10-19，Asia/Shanghai。来源：zjh-hao/Big-Data-Algorithmic-Discrimination-Experiment，基线提交5a43e3758933f3440b6c53a1dd2224407c62577d。
- 当前项目根为`/Users/pytenter/Documents/临时项目1/Big-Data-Algorithmic-Discrimination-Experiment`。独立分支feat/fairness-research-v2，克隆后初始clean。本轮只在research_extension_v2中新增文件（缺失原始数据按原来源添加到data/raw，但旧source.json不改）。没有推送或PR。
- Python3.12.14，macOS ARM CPU，4线程。核心numpy2.1.3、pandas2.2.3、scipy1.14.1、sklearn1.5.2、xgboost2.1.4、pyarrow19.0.1、matplotlib3.9.4。完整版本见requirements-local.txt与results/run_001/identity.json。
- 数据恢复：复用src/prepare_data.py的固定URL/哈希，不运行会重写source.json的main。三文件约58MB，已逐文件SHA-256验证，见results/data_recovery.json。原始Kaggle发布与第三方镜像边界保留。
- 原跟踪文件全部SHA-256在results/run_001/legacy_hashes.json；实验输入代码/配置/协议/原数据在identity.json；输出在output_hashes.json；最后交付身份另见DELIVERY_MANIFEST.json。原始数据及逐条贷款材料保留本机，不加入Git。
- 原数据读取包含原标签以重建既定分层划分，但训练只使用训练标签，校准只用校准标签，规则只用验证标签。全部六模型规则写入frozen_policies.json后才统一预测/评价测试。P2决策函数无标签参数，计划先冻结，估计函数无oracle参数。

## 随机与固定参数

划分seed20261013；树seed731；50%偏置采样seed17/29/43/61/89。200树depth5，learning_rate .05，hist、max_bin128、subsample .9、colsample_bytree .9、lambda5。4线程。共享训练词表。训练、reference、calibration、validation、test比例50/10/10/10/20。

P0风险.05、主epsilon.02、delta.01、minimum500、grid512；每组包含统一业务阈值及拒绝边界，实际最多514点。epsilon[.01,.02,.05]、delta[0,.01,.02]全部保留。P1收入训练分位数1/3、2/3，界117000/180000，无货币换算。子群n与正常标签支持阈值100。P2探索率[0,.05,.10]，五seed+offset850000、相同uniform嵌套，ESS下限100。配对bootstrap1000次，seed20261019。全部参数集中config.json。

## 实际执行命令与状态

在仓库根执行，日志位于research_extension_v2/results。为便于本机XGBoost加载已有OpenMP，以下命令的环境前缀适用于本机：

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib"
.venv/bin/python -m research_extension_v2.restore_data
.venv/bin/python -m compileall -q research_extension_v2
.venv/bin/ruff check research_extension_v2 --select F821,F822,F823
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s audit_extension -p 'test_methods.py' -v
.venv/bin/python -m unittest discover -s research_extension_v2/tests -v
.venv/bin/python -u -m research_extension_v2.run
.venv/bin/python -m research_extension_v2.verify
.venv/bin/python -m research_extension_v2.calibration_diagnostic
.venv/bin/python -m research_extension_v2.report
```

静态编译及未定义符号检查PASS；原tests10项、audit_extension3项、新数学测试17项，共30项PASS。真实run_001耗时132.75秒（6模型恢复、冻结、评价和新回放，不含下载、测试、独立核验、报告和PPT）。63可行策略/15不可行，78总行；子群90行；90台账；IPW180组行，含60 NOT_IDENTIFIABLE（零探索）。独立核验3707项PASS，未重跑历史9768项。

本机最初XGBoost缺libomp，从已有torch运行库复制到项目.venv/lib/libomp.dylib并设置上述动态库搜索路径；未修改系统或Bundled Runtime。新机器macOS可按XGBoost要求安装其OpenMP依赖。Windows一般使用发行包附带/系统对应运行库。

启动第一次因Path未导入在任何训练之前退出；保留attempt_01_startup_failure.log。修正导入后才产生run_001，未改科学定义。NumPy/sklearn发出的matmul浮点警告完整保留在execution.log；6校准器独立元素级平均梯度最大绝对值<6e-5，低于预设数值诊断1e-3阈值，所有bootstrap区间有限。这个诊断是事后实现检查，不用于选模型或规则。没有删除警告或将其当作正面科学证据。

## Windows PowerShell

完整命令见README.md。现有本机材料用run.ps1直接单测与核验，不训练；新机器先安装requirements.txt，再restore_data，再用新目录运行（例如run_002），随后verify/report显式传--run。不能覆盖run_001。完整恢复只需6模型，不需要运行原run_complete.ps1或audit_extension/run.ps1。

## PPT

使用Codex提供的Node.js与@oai/artifact-tool，不使用python-pptx。源码为presentation/build_deck.mjs，输入content.json/evidence.json，英文讲稿与中文逐页解释分别保存。总8页，946个英文口述词，时间计划总480秒（约118词/分），不是实测录音时长。

本机创建命令：

```bash
.venv/bin/python research_extension_v2/presentation/prepare_content.py
RUNTIME_NODE_MODULES=/Users/pytenter/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
/Users/pytenter/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
research_extension_v2/presentation/build_deck.mjs
```

源码需要同版本演示文稿技能与运行库（可用PRESENTATION_SKILL_DIR、RUNTIME_PYTHON环境变量指定）。裸模块import通过presentation/node_modules链接到Bundled Runtime。最终文件生成后不会覆盖；重建时应指定新输出文件名或在源码中设置新目标路径。图表使用两位小数的展示数值作为可编辑工作簿快照，精确原数保留CSV，不由展示值反推实验。

构建时修正了缺少运行库环境变量、表格核验声明、Excel图表精度及字体字段这四类导出问题，构建日志全部保留；没有改变实验结果。八页均渲染检查；原生表格与图表、嵌入工作簿、字体及结构由技能finalizer检查。未在Microsoft PowerPoint程序中实际打开测试，不声称跨应用表现已证明。

## 仍未运行

原逐条反馈台账核验：BLOCKED_NOT_RUN（公开仓库无台账/检查点）。本轮独立静态回放已运行，不能代替历史多轮闭环重建。Budgeted Exploration：NOT_RUN（可选设计），未以测试标签选rate。AWS/分布式部署：NOT_RUN，非本轮实证范围。真实拒贷反事实：现有数据不可识别。
