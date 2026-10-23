"""Frozen-score follow-up analyses for calibration, funding caps and grid size.

All threshold rules are selected on validation and saved before test replay.
This module never trains models or chooses a best configuration using test data.
Run with ``python -m research_extension_v3.followup_analysis --phase ...``.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .business import apply_policy, evaluate_policies, fit_policies, metrics_for_decisions
from .common import HERE, dump, now, sha

MODELS = ("lightgbm", "tabm")
SEEDS = (731, 732, 733)
GRIDS = (256, 513, 1025)
CAPITALS = (1.0, 0.8, 0.6, 0.4)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def inputs(run: Path, models=MODELS, seeds=SEEDS):
    """Require every requested completed job; do not silently use a partial run."""
    cfg = read_json(run / "config_frozen.json")
    jobs = []
    for model in models:
        for seed in seeds:
            name = f"{model}_s{seed}"
            path = run / "jobs" / f"{name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Required model job missing: {path}")
            job = read_json(path)
            if job.get("status") != "PASS":
                raise RuntimeError(f"Required model job is not PASS: {path}")
            paths = {s: run / "predictions" / f"{name}_{s}.parquet" for s in ("validation", "test")}
            for split, pred in paths.items():
                if not pred.exists():
                    raise FileNotFoundError(pred)
                registered = job.get("prediction_sha256", {}).get(split)
                if registered and sha(pred) != registered:
                    raise ValueError(f"Prediction hash changed: {pred}")
            jobs.append((model, int(seed), name, paths, job))
    return cfg, jobs


def provenance(run, cfg, jobs, phase):
    return dict(phase=phase, analysis_source_sha256=sha(__file__),
                business_source_sha256=sha(HERE / "business.py"),
                configuration_sha256=sha(run / "config_frozen.json"),
                predictions={name: {s: sha(p) for s, p in paths.items()}
                             for _, _, name, paths, _ in jobs},
                models=sorted({m for m, _, _, _, _ in jobs}),
                seeds=sorted({s for _, s, _, _, _ in jobs}),
                selection_protocol="validation-only threshold fitting; calibration parameters already frozen upstream",
                evaluation_protocol="test replay only; historical holdout was previously inspected",
                economic_interpretation="Scenario contribution, not observed bank profit; TARGET means repayment difficulty")


def score_metrics(frame):
    y = frame.TARGET.to_numpy()
    p = frame.probability.to_numpy()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == k) * abs(np.mean(y[bins == k]) - np.mean(p[bins == k]))
              for k in range(10) if np.any(bins == k))
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), ece_10_equal_width=float(ece))


def frame_variant(frame, calibration):
    out = frame.copy()
    if calibration == "raw":
        if "raw_probability" not in out:
            raise ValueError("Calibration ablation requires saved raw_probability")
        out["probability"] = out.raw_probability
    return out


def savefig(fig, folder, name, data, caption):
    folder.mkdir(parents=True, exist_ok=True)
    csv(folder / f"{name}.csv", data)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    dump(folder / f"{name}.json", dict(caption=caption, generated_utc=now(),
        input_csv_sha256=sha(folder / f"{name}.csv"),
        output_sha256={s: sha(folder / f"{name}.{s}") for s in ("png", "pdf")}))


def calibration_ablation(run, models=MODELS, seeds=SEEDS):
    cfg, jobs = inputs(run, models, seeds)
    folder = run / "analysis" / "calibration_ablation"
    folder.mkdir(parents=True, exist_ok=True)
    policy_rows, score_rows, differences, analytic_rows = [], [], [], []
    for model, seed, name, paths, job in jobs:
        print(f"Calibration ablation: {name}", flush=True)
        validation = pd.read_parquet(paths["validation"])
        results = {}
        for variant in ("raw", "platt"):
            val = frame_variant(validation, variant)
            frozen = fit_policies(val, cfg)
            frozen.update(model=model, seed=seed, calibration_variant=variant,
                          frozen_utc=now(), validation_input_sha256=sha(paths["validation"]),
                          business_source_sha256=sha(HERE / "business.py"))
            # No test-dependent choice is possible before this rule is persisted.
            dump(folder / "policies" / f"{name}_{variant}.json", frozen)
            test = frame_variant(pd.read_parquet(paths["test"]), variant)
            for split, frame in (("validation", val), ("test", test)):
                score_rows.append(dict(model=model, seed=seed, variant=variant, split=split,
                                       records=len(frame), **score_metrics(frame)))
            rows = evaluate_policies(test, frozen, cfg)
            for row in rows:
                row.update(model=model, seed=seed, calibration_variant=variant,
                           decision_protocol="validation_empirical_profit_quantile_grid")
            policy_rows.extend(rows)
            results[variant] = (frozen, test)
            # This fixed probability rule makes calibration economically relevant:
            # E[unit payoff] = 1 - (1+lambda)*p, so approve p <= 1/(1+lambda).
            for lam in sorted(set(map(float, cfg.get("loss_ratios", [3, 5, 7, 10])))):
                threshold = 1.0 / (1.0 + lam)
                rule = dict(model=model, seed=seed, calibration_variant=variant,
                            loss_ratio=lam, threshold=threshold,
                            selection_protocol="analytic expectation threshold; no fitted validation threshold",
                            risk_constraint_enforced=False, capital_constraint_enforced=False,
                            fairness_constraint_enforced=False, frozen_utc=now())
                dump(folder / "analytic_rules" / f"{name}_{variant}_loss{lam:g}.json", rule)
                metrics = metrics_for_decisions(test, test.probability.to_numpy() <= threshold, lam, cfg)
                analytic_rows.append({**rule, **metrics, "evaluation_split": "test",
                                      "test_primary_risk_budget_met": metrics["risk"] is not None and
                                      metrics["risk"] <= cfg["primary_risk_budget"]})
        raw_rules, raw_test = results["raw"]
        cal_rules, cal_test = results["platt"]
        cal_by_id = {p["policy_id"]: p for p in cal_rules["policies"]}
        for raw_policy in raw_rules["policies"]:
            calibrated = cal_by_id[raw_policy["policy_id"]]
            both = raw_policy["feasible"] and calibrated["feasible"]
            val_changed = test_changed = None
            if both:
                val_changed = int(np.count_nonzero(
                    apply_policy(frame_variant(validation, "raw"), raw_policy) !=
                    apply_policy(frame_variant(validation, "platt"), calibrated)))
                test_changed = int(np.count_nonzero(apply_policy(raw_test, raw_policy) !=
                                                  apply_policy(cal_test, calibrated)))
            differences.append(dict(model=model, seed=seed, policy_id=raw_policy["policy_id"],
                family=raw_policy["family"], utility_basis=raw_policy["utility_basis"],
                epsilon=raw_policy["epsilon"], noharm_delta=raw_policy["noharm_delta"],
                raw_feasible=raw_policy["feasible"], platt_feasible=calibrated["feasible"],
                validation_changed_approvals=val_changed, test_changed_approvals=test_changed,
                explanation="Strictly increasing calibration usually preserves rank/grid decisions; numeric ties and interpolation can differ"))
    scores = csv(folder / "score_metrics.csv", score_rows)
    policies = csv(folder / "policy_metrics.csv", policy_rows)
    diffs = csv(folder / "decision_differences.csv", differences)
    analytic = csv(folder / "analytic_threshold_metrics.csv", analytic_rows)
    primary = analytic[np.isclose(analytic.loss_ratio, cfg["primary_loss_ratio"])]
    plot_rows = scores[scores.split.eq("test")].merge(primary[["model", "seed", "calibration_variant",
        "utility_per_1000_applicants", "risk"]], left_on=["model", "seed", "variant"],
        right_on=["model", "seed", "calibration_variant"], validate="one_to_one")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    x = np.arange(len(models))
    for j, variant in enumerate(("raw", "platt")):
        group = plot_rows[plot_rows.variant.eq(variant)].groupby("model").mean(numeric_only=True).reindex(models)
        for ax, col, factor, title in zip(axes, ("brier", "utility_per_1000_applicants", "risk"),
                (1, 1, 100), ("Test probability quality", "Analytic profit rule", "Analytic rule risk")):
            ax.bar(x + (j - .5) * .35, group[col] * factor, width=.35, label=variant)
            ax.set_title(title)
            ax.set_xticks(x, models)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[1].set_ylabel("Scenario contribution / 1,000")
    axes[2].set_ylabel("Difficulty labels among approved (%)")
    axes[2].axhline(cfg["primary_risk_budget"] * 100, color="grey", ls="--", lw=1)
    axes[0].legend()
    fig.suptitle("Calibration ablation: fixed probability rule has no hard risk cap")
    fig.tight_layout()
    savefig(fig, folder / "figures", "calibration_business", plot_rows,
            "Raw vs frozen Platt scores. Analytic threshold p<=1/(1+lambda) is not a risk-constrained method. Mean across completed requested seeds.")
    proof = provenance(run, cfg, jobs, "calibration_ablation")
    proof.update(status="PASS", completed_utc=now(), score_records=len(scores),
        policy_records=len(policies), analytic_records=len(analytic),
        feasible_paired_rules=int((diffs.raw_feasible & diffs.platt_feasible).sum()),
        test_identical_paired_rules=int(diffs.test_changed_approvals.eq(0).sum()),
        interpretation="No forced improvement: monotone recalibration can leave empirical-profit threshold policies unchanged while changing a probability-based analytic rule.",
        hashes={p.name: sha(p) for p in folder.glob("*.csv")})
    dump(folder / "completion.json", proof)
    print(json.dumps({k: proof[k] for k in ("status", "policy_records", "analytic_records")}), flush=True)


def grid_capital(run, models=MODELS, seeds=SEEDS):
    cfg, jobs = inputs(run, models, seeds)
    folder = run / "analysis" / "grid_capital"
    folder.mkdir(parents=True, exist_ok=True)
    rows, validation_rows = [], []
    for model, seed, name, paths, job in jobs:
        val = pd.read_parquet(paths["validation"])
        for grid in GRIDS:
            print(f"Grid/capital sensitivity: {name}, grid={grid}", flush=True)
            setting = copy.deepcopy(cfg)
            setting.update(grid=grid, capital_fractions=list(CAPITALS), epsilons=[.02, .01],
                           noharm_deltas=[.01])
            # Main grid examines risk and loss one parameter at a time; other
            # grids isolate resolution/capital rather than their full Cartesian product.
            setting["risk_budgets"] = [.03, .05, .08] if grid == 513 else [cfg["primary_risk_budget"]]
            setting["loss_ratios"] = [3, 5, 7, 10] if grid == 513 else [cfg["primary_loss_ratio"]]
            frozen = fit_policies(val, setting)
            frozen.update(model=model, seed=seed, frozen_utc=now(),
                          validation_input_sha256=sha(paths["validation"]), analysis_config=setting,
                          business_source_sha256=sha(HERE / "business.py"))
            policy_path = folder / "policies" / f"{name}_grid{grid}.json"
            dump(policy_path, frozen)
            test = pd.read_parquet(paths["test"])
            for row in evaluate_policies(test, frozen, setting):
                row.update(model=model, seed=seed, requested_grid=grid,
                           frozen_policy_sha256=sha(policy_path),
                           analysis="grid_capital_sensitivity")
                rows.append(row)
            for policy in frozen["policies"]:
                vm = policy.get("validation_metrics") or {}
                validation_rows.append({"model": model, "seed": seed, "requested_grid": grid,
                    "policy_id": policy["policy_id"], "family": policy["family"],
                    "feasible": policy["feasible"], "capital_fraction": policy["capital_fraction"],
                    "loss_ratio": policy["loss_ratio"], "risk_budget": policy["risk_budget"],
                    "epsilon": policy["epsilon"], "noharm_delta": policy["noharm_delta"],
                    "grid_actual": frozen["metadata"]["grid_actual"], **vm})
    frame = csv(folder / "test_policy_metrics.csv", rows)
    csv(folder / "validation_policy_metrics.csv", validation_rows)
    amount = frame[frame.feasible.eq(True) & frame.utility_basis.eq("amount_utility") &
                   frame.family.eq("group_profit_gap") & np.isclose(frame.epsilon.fillna(-1), .02)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (model, grid), part in amount.groupby(["model", "requested_grid"], sort=False):
        mean = part.groupby("capital_fraction")[["amount_utility_per_1000_applicants", "abs_frr_gap"]].mean()
        for ax, column, factor in zip(axes, ("amount_utility_per_1000_applicants", "abs_frr_gap"), (1e-6, 100)):
            ax.plot(mean.index, mean[column] * factor, "o-", label=f"{model}, grid {grid}")
    axes[0].set(xlabel="Validation funding cap / requested capital", ylabel="Scenario amount contribution / 1,000 (million source units)",
                title="Amount payoff across funding caps")
    axes[1].set(xlabel="Validation funding cap / requested capital", ylabel="Test absolute FRR gap (pp)",
                title="Test fairness after frozen validation choices")
    axes[1].axhline(2, color="grey", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Grid and capital sensitivity; requested grids are not all nested")
    fig.tight_layout()
    savefig(fig, folder / "figures", "grid_capital_business", amount,
            "Mean across requested seeds; same frozen scores, validation-only rules. Requested grids 256/513/1025, funding caps 1/.8/.6/.4. No monotonic grid-optimum claim for nonnested grids. Test risk/capital/gap breaches remain in source CSV.")
    proof = provenance(run, cfg, jobs, "grid_capital")
    proof.update(status="PASS", completed_utc=now(), policy_records=len(frame),
                 grids=list(GRIDS), capital_fractions=list(CAPITALS),
                 infeasible_records=int((~frame.feasible).sum()),
                 sweep_protocol="Grid x capital at primary risk/loss; risk/loss one-at-a-time only at grid513",
                 hashes={p.name: sha(p) for p in folder.glob("*.csv")})
    dump(folder / "completion.json", proof)
    print(json.dumps({k: proof[k] for k in ("status", "policy_records", "infeasible_records")}), flush=True)


def primary_rows(run, cfg):
    frame = pd.read_csv(run / "policy_metrics.csv")
    mask = frame.utility_basis.eq("unit_utility") & frame.family.isin(
        ["group_profit", "group_profit_gap", "group_profit_gap_noharm"])
    mask &= np.isclose(frame.loss_ratio, cfg["primary_loss_ratio"])
    mask &= np.isclose(frame.risk_budget, cfg["primary_risk_budget"])
    mask &= np.isclose(frame.capital_fraction, 1)
    mask &= frame.family.eq("group_profit") | np.isclose(frame.epsilon.fillna(-1), cfg["primary_epsilon"])
    mask &= frame.family.ne("group_profit_gap_noharm") | np.isclose(frame.noharm_delta.fillna(-1), .01)
    return frame[mask].copy()


def markdown_table(frame):
    if frame.empty:
        return "（没有可用记录）"
    def value(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        if isinstance(x, (float, np.floating)):
            return f"{x:.5f}"
        return str(x).replace("|", "/")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def finalize(run):
    cfg, jobs = inputs(run)
    for phase in ("calibration_ablation", "grid_capital"):
        phase_folder = run / "analysis" / phase
        proof = read_json(phase_folder / "completion.json")
        if proof.get("status") != "PASS" or proof.get("models") != sorted(MODELS) or proof.get("seeds") != list(SEEDS):
            raise ValueError(f"Follow-up analysis incomplete: {phase}")
        current = provenance(run, cfg, jobs, phase)
        for key in ("predictions", "configuration_sha256", "business_source_sha256", "analysis_source_sha256"):
            if proof[key] != current[key]:
                raise ValueError(f"Analysis input/source changed after completion: {phase}/{key}")
        for name, digest in proof.get("hashes", {}).items():
            if sha(phase_folder / name) != digest:
                raise ValueError(f"Analysis result changed after completion: {phase}/{name}")
    old = HERE / "results" / "run_001"
    old_cfg = read_json(old / "config_frozen.json")
    if sha(run / "split_manifest.parquet") != sha(old / "split_manifest.parquet"):
        raise ValueError("run001 and run002 use different population/split manifests")
    folder = run / "analysis" / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    model_frames, policy_frames = [], []
    for label, directory, config in (("run_001", old, old_cfg), ("run_002", run, cfg)):
        mm = pd.read_csv(directory / "model_metrics.csv")
        mm["run"] = label
        model_frames.append(mm)
        pp = primary_rows(directory, config)
        pp["run"] = label
        policy_frames.append(pp)
    models = csv(folder / "model_seed_comparison.csv", pd.concat(model_frames, ignore_index=True))
    policies = csv(folder / "primary_policy_comparison.csv", pd.concat(policy_frames, ignore_index=True))
    model_summary = models.groupby(["run", "model"]).agg(
        seeds=("seed", "nunique"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
        brier_mean=("brier", "mean"), reference_auc_mean=("reference_auc", "mean")).reset_index()
    csv(folder / "model_summary.csv", model_summary)
    feasible = policies[policies.feasible.eq(True)].copy()
    summaries = feasible.groupby(["run", "model", "family"]).agg(
        seeds=("seed", "nunique"), utility=("utility", "mean"),
        utility_per_1000=("utility_per_1000_applicants", "mean"),
        approval_rate=("approval_rate", "mean"), risk=("risk", "mean"),
        thin_frr=("thin_frr", "mean"), thick_frr=("thick_frr", "mean"),
        abs_frr_gap=("abs_frr_gap", "mean")).reset_index()
    csv(folder / "policy_summary.csv", summaries)
    pair_keys = ["run", "model", "seed", "risk_budget", "loss_ratio", "capital_fraction"]
    baseline = feasible[feasible.family.eq("group_profit")]
    candidate = feasible[feasible.family.isin(["group_profit_gap", "group_profit_gap_noharm"])]
    pairs = candidate.merge(baseline, on=pair_keys, suffixes=("_candidate", "_baseline"), validate="many_to_one")
    pairs["utility_difference"] = pairs.utility_candidate - pairs.utility_baseline
    pairs["utility_change_pct"] = np.where(pairs.utility_baseline > 0,
        100 * pairs.utility_difference / pairs.utility_baseline, np.nan)
    pairs["thin_frr_change_pp"] = 100 * (pairs.thin_frr_candidate - pairs.thin_frr_baseline)
    pairs["thick_frr_change_pp"] = 100 * (pairs.thick_frr_candidate - pairs.thick_frr_baseline)
    csv(folder / "matched_fairness_cost.csv", pairs)
    calibration = pd.read_csv(run / "analysis/calibration_ablation/decision_differences.csv")
    analytic = pd.read_csv(run / "analysis/calibration_ablation/analytic_threshold_metrics.csv")
    grid = pd.read_csv(run / "analysis/grid_capital/test_policy_metrics.csv")
    cols = ["run", "model", "family", "seeds", "utility_per_1000", "approval_rate", "risk", "thin_frr", "thick_frr", "abs_frr_gap"]
    body = ["# 后续实验：强模型、公平成本与业务敏感性\n",
        "本报告由已经保存的真实结果生成。run_001 是首轮基线，run_002 是事先排定的扩大调参和三种子复现。所有预测器参数由 reference 选择，Platt 参数由 calibration 拟合，审批阈值由 validation 选择后冻结；test 只重放。历史留出集此前已查看，不宣称新的盲测。",
        "业务口径：单位净贡献=正常标签获批数−5×困难标签获批数；主风险预算5%，公平容差2个百分点，群体保护允许相对同模型无公平约束收益基准恶化最多1个百分点。TARGET=1 表示还款困难，金额与损失均为业务情景；金额单位保留数据原始单位。",
        "## 统一人群下的模型比较\n\n" + markdown_table(model_summary),
        "模型使用同一划分和相同原始可用信息，处理方式遵循各模型适配。扩大调参的候选配置与 reference 分数保存在 run_002/tuning；表中列出真实表现，包括调参后没有改善的情况。论文方法在统一 Home Credit 场景的适配结果不能称为原论文所有基准复现，也不能据此宣称 SOTA。",
        "## 同一业务口径的审批结果\n\n" + markdown_table(summaries[cols]),
        "同模型、同种子 group_profit 与公平变体的配对结果见 analysis/comparison/matched_fairness_cost.csv。跨模型差异同时包含预测能力变化；收益应和放贷规模、风险、两组误拒变化共同解读。不可行策略保留在原表，不用可行均值掩盖缺失。"]
    for (label, model, family), part in pairs.groupby(["run", "model", "family_candidate"]):
        body.append(f"{label} / {model} / {family}：{len(part)} 个配对种子的平均净贡献变化 {part.utility_difference.mean():+.2f}（{part.utility_change_pct.mean():+.3f}%），薄信用记录组误拒率变化 {part.thin_frr_change_pp.mean():+.3f} 个百分点，对照组变化 {part.thick_frr_change_pp.mean():+.3f} 个百分点。")
    paired = calibration[calibration.raw_feasible.eq(True) & calibration.platt_feasible.eq(True)]
    identical = int(paired.test_changed_approvals.eq(0).sum())
    body += ["## 校准消融\n",
        f"raw 与 Platt 在 {len(paired)} 对可行的验证经验收益审批规则中，有 {identical} 对测试审批完全相同。严格单调校准通常保留排序；在分位数候选与经验标签收益搜索中，出现零差异是合理结果。代码逐人比较决定并保留变化数量，没有把概率改善自动解释为业务改善。",
        "另列解析期望收益规则 p≤1/(1+lambda)。该规则直接使用概率值，因此可以检验校准对业务决策的影响。解析规则没有施加5%风险预算，也没有资金或公平硬约束，不能直接当作已经满足相同风险要求的竞争方法。概率 Brier/log-loss/ECE 与解析规则风险、净贡献见 analysis/calibration_ablation/。",
        "![Calibration ablation](analysis/calibration_ablation/figures/calibration_business.png)",
        "## 资金与网格敏感性\n",
        "固定模型分数后，在256、513、1025三个请求网格和100%/80%/60%/40%请求本金预算下重做验证优化；513网格另外执行风险3%/5%/8%与损失比3/5/7/10的单因素实验。网格不全嵌套，不能声称网格变密必然单调改进。验证冻结策略和测试逐策略数据分别保存。",
        f"资金/网格分析共 {len(grid)} 条策略记录，其中 {int((~grid.feasible.eq(True)).sum())} 条不可行。验证条件满足并不保证测试条件满足；测试风险、资金、公平和群体保护的结果均保留在 CSV。",
        "![Funding and grid sensitivity](analysis/grid_capital/figures/grid_capital_business.png)"]
    for label, frame in policies.groupby("run"):
        valid = frame[frame.feasible.eq(True)]
        constrained = valid[valid.epsilon.notna()]
        body.append(f"{label} 主比较表：{len(valid)} 条可行记录；测试风险超预算 {int((valid.risk > valid.risk_budget + 1e-10).sum())} 条；带公平要求的记录中测试差距超容差 {int((constrained.abs_frr_gap > constrained.epsilon + 1e-10).sum())} 条。")
    stress = run / "bias_stress" / "policy_metrics.csv"
    if stress.exists():
        s = pd.read_csv(stress)
        body.append(f"强模型偏见压力测试另有 {len(s)} 条策略记录，位于 bias_stress/；采样种子与模型种子是不同变量，应结合其任务元数据解释。该部分考察原始选择偏差及修正方式在较强预测器上的迁移情况。")
    body += ["## 可用于汇报的结论组织\n",
        "先用训练数据实验解释偏见来源，再用同模型公平成本说明新增获批与新增误拒各贡献了多少净收益，最后比较强模型能否覆盖公平代价。优先展示收益—误拒差距前沿、放贷规模与风险、群体误拒变化。若方法没有超过强基线，按实际指标呈现其适用条件和权衡。",
        "所有图区分验证选择与测试评价，图片来自实际 CSV；PNG/PDF、图源和校验信息保存于对应 figures/。本阶段不改写 run_001 结果。"]
    text = "\n\n".join(body) + "\n"
    (run / "FOLLOWUP_RESULTS_CN.md").write_text(text, encoding="utf-8")
    (run / "FOLLOWUP_RESULTS_CN.txt").write_text(text, encoding="utf-8")
    proof = provenance(run, cfg, jobs, "finalize")
    proof.update(status="PASS", completed_utc=now(),
                 baseline_split_sha256=sha(old / "split_manifest.parquet"),
                 model_seed_records=len(models), primary_policy_records=len(policies),
                 matched_fairness_cost_records=len(pairs),
                 report_sha256=sha(run / "FOLLOWUP_RESULTS_CN.md"),
                 hashes={p.name: sha(p) for p in folder.glob("*.csv")})
    dump(folder / "completion.json", proof)
    print(json.dumps(dict(status="PASS", report=str(run / "FOLLOWUP_RESULTS_CN.md")), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("calibration_ablation", "grid_capital", "finalize"))
    parser.add_argument("--run", type=Path, default=HERE / "results" / "run_002")
    parser.add_argument("--models", default=",".join(MODELS), help="Subset for an explicit smoke run; finalize requires all models")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="Subset for an explicit smoke run; finalize requires all three seeds")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.phase == "finalize":
        finalize(run)
    else:
        models = tuple(args.models.split(","))
        seeds = tuple(int(x) for x in args.seeds.split(","))
        if not models or not seeds:
            raise ValueError("At least one model and seed must be requested")
        globals()[args.phase](run, models, seeds)


if __name__ == "__main__":
    main()
