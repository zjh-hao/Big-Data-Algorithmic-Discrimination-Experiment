# 审查问题 1、2、4 的补充实验

此目录独立保存新实验，使用上级目录的 Home Credit 数据、划分和 Python 环境，不覆盖旧模型或报告。全部在本机CPU、4线程运行。

## 阅读顺序

1. [RESULTS_REPORT.md](RESULTS_REPORT.md)：完整结果、对照解释、限制与汇报建议。
2. [PROTOCOL.md](PROTOCOL.md)：运行前固定的设计与损失函数公式。
3. [PRESENTATION_NOTES.md](PRESENTATION_NOTES.md)：汇报摘要。

## 新增实验

| 模块 | 唯一新增模型 | 对照 |
|---|---:|---|
| A 采样机制 | 41 | 同总量随机、同组人数且保持组内标签比例、标签失真、纠正，2比例×5种子＋完整基线 |
| B 训练损失 | 24 | 5种交叉熵目标×6数据条件，复用6个A基线，30行结果 |
| C 反馈信息 | 155 | 3种验证来源×2获取策略×5轮×5种子＋5共同初始模型 |

合计220个唯一模型；额外450行同模型验证信息诊断不另训练模型。所有新静态实验也统一200棵树，不与旧400棵结果直接相减。

机会损失通过训练权重直接改变加权交叉熵：它给予两组正常标签的平均损失等大组权重，属于公平代理目标，不能保证FRR差距达到某个约束。代码不会根据测试结果挑选损失系数或轮次。

反馈所有记录预先指定训练/校准/验证角色；只用已经揭示的标签。external模式使用与本路径已揭示保留记录数匹配的独立池；observed模式只使用可见反馈；frozen只使用初始保留信息。每批标签假定在下一次更新前已成熟。

## 运行

在本目录 PowerShell：

```powershell
./run.ps1
```

脚本显式启用Python UTF-8模式，以支持中文项目路径下的身份记录和断点恢复。

依赖沿用 `../requirements-lock.txt` 与 `../.venv`，不需GPU或AWS账户。脚本运行单测、训练、独立核验和报告。完整结果存在时只核验并重新生成报告。训练中断时可复用已完成模型和反馈种子检查点；若冻结输入发生变化会拒绝混用结果，需新建结果目录。

单独刷新报告：

```powershell
../.venv/Scripts/python.exe -X utf8 verify.py
../.venv/Scripts/python.exe -X utf8 report.py
```

## 文件与证据

- `config.json`：固定模型、种子、损失系数。
- `run_experiment.py`：采样、训练、回放、冻结后评价。
- `test_methods.py`：抽样、损失等价性、信息来源方法测试。
- `verify.py`：从保存台账独立核验来源、权重、阈值和指标。
- `results/identity.json`：原数据、代码、协议、词表、划分哈希。
- `results/frozen_all.json`：训练完毕、测试评估开始前冻结的所有策略。
- `results/training/`、`policies/`：各模型训练ID/权重与校准/验证ID。
- `results/ledgers/`：按批次决策、探索概率与标签可见性。
- `results/metrics.csv`：全部701行策略评价；`metrics_A/B/C/C_shadow.csv`分模块。
- `results/paired_intervals.csv`：全部配对差值区间。
- `results/stream_metrics.csv`：包含探索的实际批准风险与人数。
- `results/verification.json`：核验结果；`execution.log`：训练日志。
- `models/`：220个CPU模型与预处理器；`figures/`：6组PNG/PDF图。

## 仍然保留的边界

这是既有历史样本上的追加机制实验，旧测试结果已查看。分组差距不自动证明歧视，未知拒贷反事实和真实日历时间未解决；既有贷款ID不保证自然人不跨集合。独立验证、机会权重和探索各有条件与代价。5%是验证样本经验预算，不能保证探索后整体风险或未来风险。
