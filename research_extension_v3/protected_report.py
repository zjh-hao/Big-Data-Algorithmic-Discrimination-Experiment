"""Report saved profit-budgeted, subgroup-protected approval experiments.

This module never fits a predictor or selects a policy from test outcomes. Each
figure exports its source rows. Comparisons of costs match the LP profit baseline
within model/seed/scenario; deterministic rules remain a distinct decision family.
Run: python -m research_extension_v3.protected_report --run <run directory>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
COLORS = ["#1B7373", "#CC7435", "#4F68A3", "#8E5E9C", "#6A8B42", "#C95B6A"]
SUBGROUPS = [f"subgroup_{g}_{i}_frr" for g in (0, 1) for i in (0, 1, 2)]
SUBGROUP_LABELS = ["Thick / low", "Thick / middle", "Thick / high",
                   "Thin / low", "Thin / middle", "Thin / high"]
ROLE_LABELS = {
    "lp_profit_baseline": "LP profit baseline",
    "lp_gap": "LP profit + gap",
    "lp_gap_macro_protected": "LP gap + macro protection",
    "minimax_macro": "Six-group minimax / macro protection",
    "minimax_six": "Six-group minimax / six-group protection",
    "minimax_six_robust": "Two-fold constrained six-group minimax",
    "adaptive_primary": "Minimum-harm adaptive six-group minimax",
    "adaptive_robust": "Two-fold adaptive six-group minimax",
    "lp_profit_risk_robust": "LP profit / two-fold risk constraints",
    "lp_gap_robust": "LP gap / two-fold risk and gap constraints",
}


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_csv(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["source_file"] = str(path.resolve())
    frame["source_row"] = np.arange(len(frame)) + 2
    return frame


def bools(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "pass"])


def table(frame, digits=4):
    if frame.empty:
        return "（暂无满足该预设比较的记录。）"
    clean = frame.copy()
    for column in clean:
        if pd.api.types.is_numeric_dtype(clean[column]):
            clean[column] = clean[column].map(lambda x: "NA" if pd.isna(x) else f"{x:.{digits}f}")
    clean = clean.fillna("NA").astype(str).map(lambda x: x.replace("|", "/"))
    lines = ["| " + " | ".join(clean.columns) + " |",
             "| " + " | ".join(["---"] * len(clean.columns)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in clean.itertuples(index=False, name=None))
    return "\n".join(lines)


def normalize(frame):
    if frame.empty:
        return frame
    out = frame.copy()
    # Runner retains validation and test prefices to make the split explicit.
    # Read test metrics into short report aliases, keeping both source columns.
    for column in list(out):
        if column.startswith("test_") and column[5:] not in out:
            suffix = column[5:]
            if not suffix.endswith("_met") and suffix != "relative_profit_change":
                out[suffix] = out[column]
    aliases = {"policy": "policy_id", "family": "policy_family", "approved_risk": "risk",
               "delta": "noharm_delta", "profit_budget": "beta", "profit_loss_budget": "beta"}
    for old, new in aliases.items():
        if new not in out and old in out:
            out[new] = out[old]
    if "policy_id" not in out:
        raise ValueError("Saved policy metrics must contain policy_id or policy")
    if "experiment" not in out:
        out["experiment"] = "main"
    if "robust" not in out:
        out["robust"] = out.policy_id.astype(str).str.contains("robust")
    else:
        out["robust"] = bools(out.robust)
    if "feasible" not in out:
        out["feasible"] = out.get("status", pd.Series("", index=out.index)).astype(str).str.upper().isin(
            ["PASS", "OK", "OPTIMAL", "FEASIBLE", "SUCCESS"])
    else:
        out["feasible"] = bools(out.feasible)
    for col in ["beta", "epsilon", "noharm_delta", "risk_budget", "capital_fraction", "loss_ratio"]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "baseline_policy_id" not in out:
        out["baseline_policy_id"] = "lp_profit_baseline"
    if "mode" not in out:
        out["mode"] = "unknown"
    if "protection" not in out:
        out["protection"] = "none"
    if "minimax_scope" not in out:
        out["minimax_scope"] = "none"
    flag_aliases = {"test_risk_met": "test_risk_constraint_met",
                    "test_capital_met": "test_capital_constraint_met",
                    "test_gap_met": "test_gap_constraint_met"}
    for source, target in flag_aliases.items():
        if target not in out and source in out:
            out[target] = out[source]
    if "test_noharm_constraint_met" not in out:
        out["test_noharm_constraint_met"] = pd.Series(None, index=out.index, dtype=object)
        for protection, source in [("macro", "test_macro_protection_met"), ("six", "test_six_protection_met")]:
            if source in out:
                take = out.protection.eq(protection)
                out.loc[take, "test_noharm_constraint_met"] = out.loc[take, source]

    def role(row):
        policy_id = str(row.policy_id)
        if policy_id in ROLE_LABELS:
            return policy_id
        if policy_id.startswith("adaptive_"):
            return "adaptive_robust" if row.robust else "adaptive_primary"
        if policy_id == "lp_profit_baseline":
            return "lp_profit_baseline"
        if str(row.get("mode")) == "budgeted_minimax":
            # In the registered ablation, both variants minimize worst-six FRR;
            # macro versus six names indicate protection scope, not objective.
            name = "minimax_macro" if str(row.get("protection")) == "macro" else "minimax_six"
            return name + ("_robust" if row.robust else "")
        if str(row.get("mode")) == "profit_gap_protected":
            return "lp_gap_macro_protected" if str(row.get("protection")) == "macro" else "lp_gap_six_protected"
        if str(row.get("mode")) == "profit_gap":
            return "lp_gap"
        # Explicit IDs are retained for deterministic comparators and new ablations.
        for known in sorted(ROLE_LABELS, key=len, reverse=True):
            if policy_id == known or policy_id.startswith(known + "_"):
                return known
        return policy_id

    out["method_role"] = out.apply(role, axis=1)
    if "decision_family" not in out:
        out["decision_family"] = np.where(
            out.method_role.str.startswith(("lp_", "minimax_")),
            "group_threshold_mixture", "deterministic_two_thresholds")
    groups = [c for c in SUBGROUPS if c in out]
    if "worst_subgroup_frr" not in out and groups:
        out["worst_subgroup_frr"] = out[groups].max(axis=1)
    if "utility_per_1000_applicants" not in out and {"records", "utility"}.issubset(out):
        out["utility_per_1000_applicants"] = out.utility * 1000 / out.records
    return out


def same_lp_pairs(frame):
    """Pair saved outcomes with the identical-scenario LP baseline, never a winner."""
    if frame.empty or "utility" not in frame:
        return pd.DataFrame()
    feasible = frame[frame.feasible & frame.utility.notna() &
                     frame.decision_family.eq("group_threshold_mixture")].copy()
    base = feasible[feasible.method_role.eq("lp_profit_baseline")].copy()
    if base.empty:
        return pd.DataFrame()
    keys = [c for c in ["experiment", "model", "seed", "loss_ratio", "risk_budget",
                        "capital_fraction", "sampling_condition", "sampling_seed"] if c in frame]
    metrics = [c for c in ["utility", "amount_utility", "approval_rate", "risk", "abs_frr_gap",
                          "thin_frr", "thick_frr", "worst_subgroup_frr"] + SUBGROUPS if c in base]
    # A baseline row can be repeated for multiple sweeps; identical values are safe
    # to collapse. Distinct baselines require an explicit scenario key, never pick one.
    baseline = base[keys + ["policy_id"] + metrics].drop_duplicates()
    if baseline.duplicated(keys).any():
        raise ValueError("Ambiguous LP baselines: add the missing scenario key instead of test-selecting a row")
    baseline = baseline.rename(columns={c: "lp_baseline_" + c for c in ["policy_id"] + metrics})
    pairs = feasible.merge(baseline, on=keys, how="left", validate="many_to_one")
    pairs = pairs[pairs.lp_baseline_utility.notna()].copy()
    pairs["utility_change_vs_lp"] = pairs.utility - pairs.lp_baseline_utility
    denominator = pairs.lp_baseline_utility.where(pairs.lp_baseline_utility > 0)
    pairs["test_profit_retention"] = pairs.utility / denominator
    for metric in ["thin_frr", "thick_frr", "abs_frr_gap", "worst_subgroup_frr"] + SUBGROUPS:
        if metric in pairs and "lp_baseline_" + metric in pairs:
            pairs[metric + "_change_pp"] = 100 * (pairs[metric] - pairs["lp_baseline_" + metric])
    return pairs


def constraint_rates(frame):
    if frame.empty:
        return pd.DataFrame()
    keys = [c for c in ["model", "method_role", "robust", "beta"] if c in frame]
    rates = []
    for combo, subset in frame.groupby(keys, dropna=False, sort=False):
        combo = combo if isinstance(combo, tuple) else (combo,)
        base = dict(zip(keys, combo))
        for field, name in [("test_risk_constraint_met", "risk"),
                            ("test_capital_constraint_met", "capital"),
                            ("test_gap_constraint_met", "gap"),
                            ("test_noharm_constraint_met", "protection"),
                            ("test_profit_floor_met", "profit floor")]:
            if field not in subset:
                continue
            available = subset[field].notna()
            if name == "gap":
                available &= (subset["mode"].isin(["profit_gap", "profit_gap_protected", "budgeted_minimax"]) |
                              subset.policy_id.str.startswith("adaptive_"))
            elif name == "protection":
                available &= subset["protection"].isin(["macro", "six"])
            elif name == "profit floor":
                available &= (subset["mode"].eq("budgeted_minimax") |
                              subset.policy_id.str.startswith("adaptive_")) & subset.beta.notna()
            if not available.any():
                continue
            violated = int((~bools(subset.loc[available, field])).sum())
            rates.append({**base, "constraint": name, "evaluated": int(available.sum()),
                          "violations": violated, "violation_rate": violated / available.sum()})
    return pd.DataFrame(rates)


def primary_scenario(frame, config, include_beta=True):
    """Predeclared settings only; retain not-applicable fields, never rank test scores."""
    if frame.empty:
        return frame
    keep = pd.Series(True, index=frame.index)
    values = {"epsilon": config.get("primary_epsilon", .02),
              "noharm_delta": config.get("primary_delta", .01),
              "risk_budget": config.get("primary_risk_budget", .05),
              "loss_ratio": config.get("primary_loss_ratio", 5),
              "capital_fraction": config.get("primary_capital_fraction", 1.)}
    if include_beta:
        values["beta"] = config.get("primary_beta", .01)
    for column, value in values.items():
        if column in frame:
            appropriate = frame[column].isna() | np.isclose(frame[column], value)
            if column == "noharm_delta":
                # Adaptive protection derives an effective delta from validation;
                # it must not be excluded because the original 1pp is infeasible.
                adaptive = frame.policy_id.str.startswith("adaptive_")
                if "requested_delta" in frame:
                    appropriate |= adaptive & (frame.requested_delta.isna() | np.isclose(frame.requested_delta, value))
                else:
                    appropriate |= adaptive
            keep &= appropriate
    names = config.get("report_primary_policy_ids")
    if names and include_beta:
        keep &= frame.policy_id.isin(names)
    return frame[keep].copy()


def main(run_dir, require_complete=False):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = next((p for p in [run_dir / "config_frozen.json", run_dir / "experiments_config.json",
                                   run_dir / "config.json"] if p.exists()), None)
    config = read_json(config_path, {}) if config_path else {}
    frame = normalize(load_csv(run_dir / "policy_metrics.csv"))
    figures, sources = run_dir / "figures", run_dir / "figure_data"
    figures.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    manifest = {"source": str(run_dir / "policy_metrics.csv"),
                "input_sha256": sha(run_dir / "policy_metrics.csv") if (run_dir / "policy_metrics.csv").exists() else None,
                "configuration_sha256": sha(config_path) if config_path else None,
                "script_sha256": sha(__file__), "figures": [], "skipped": [],
                "protocol": "Frozen validation-only choices; descriptive test replay, no test winner selection"}
    manifest["additional_input_sha256"] = {
        name: sha(run_dir / name) for name in ["standard_method_comparison.csv", "standard_method_paired_intervals.csv",
                                              "paired_intervals.csv", "randomization_replays.csv"]
        if (run_dir / name).exists()}
    expected = config.get("report_expected_jobs", [])
    observed = set()
    if not frame.empty and {"model", "seed"}.issubset(frame):
        observed = (set(frame.job.astype(str)) if "job" in frame else
                    {f"{r.model}_s{int(r.seed)}" for r in frame[["model", "seed"]].drop_duplicates().itertuples()})
    missing = sorted(set(expected) - observed)
    manifest["expected_jobs"] = expected
    manifest["observed_jobs"] = sorted(observed)
    manifest["missing_jobs"] = missing
    manifest["final_complete"] = bool(expected) and not missing
    if require_complete and (not expected or missing):
        raise RuntimeError(f"Final report requires explicit report_expected_jobs and every job; missing={missing}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18,
                         "savefig.dpi": 230, "pdf.fonttype": 42})

    def save(fig, name, data, caption):
        files = [figures / f"protected_{name}.{suffix}" for suffix in ("png", "pdf")]
        source = sources / f"protected_{name}.csv"
        data.to_csv(source, index=False, encoding="utf-8-sig")
        for path in files:
            fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        manifest["figures"].append({"name": name, "caption": caption, "source_rows": len(data),
                                    "sha256": {str(p.relative_to(run_dir)): sha(p) for p in files + [source]}})

    def skip(name, reason):
        manifest["skipped"].append({"name": name, "reason": reason})

    sections = ["# 收益预算、群体保护与六子群最差误拒率：方法扩展实验", "",
        "本报告只读取已冻结策略和测试重放结果。预测器保持原实验训练版本；新增内容是审批优化。"
        "本轮方法定位是课程项目中的业务、公平与群体保护联合设计，不宣称提出新的 SOTA 算法。", "",
        "## 数据与比较口径", "",
        "单位净贡献为正常标签获批数减去损失比乘以困难标签获批数；金额净贡献来自既定情景参数。"
        "TARGET=1 表示还款困难，指标不是实际银行账面利润。策略在 validation 上拟合后冻结；"
        "test 只重放。这个历史测试集曾被查看，不能称为新的盲测。", "",
        "LP 策略对每个人给出获批概率，是单调分组阈值的随机混合。获批数、净贡献及误拒率表示期望值；"
        "风险是期望困难获批数除以期望总获批数，不是任一次随机执行都满足预算的保证。", "",
        "原两组硬阈值与 LP 混合使用不同决策空间。LP 内的公平成本必须相对相同模型、种子、"
        "损失、风险与资金情景下的 LP 无公平约束收益基准计算；不能把决策空间和阈值网格变化带来的差异计作公平方法的单独贡献。", "",
        "鲁棒变体在按申请 ID 固定分成的两个验证子集上同时施加相应约束。这是有限的验证稳定性检查，"
        "不是分布无关保证；下表保留测试超限情况。", "",
        f"主设置：风险预算 {config.get('primary_risk_budget', .05):.0%}，"
        f"公平容差 {config.get('primary_epsilon', .02) * 100:g} 个百分点，"
        f"原始严格保护目标 {config.get('primary_delta', .01) * 100:g} 个百分点，"
        f"收益预算 {config.get('primary_beta', .01):.0%}。"
        "自适应方法的有效保护上限由验证 LP 计算并另外报告，不能自动解释为满足原严格目标。", "",
        "## 覆盖与可行性", "",
        f"登记需要的模型任务：{len(expected)}；表中已出现：{len(observed)}；缺失：{', '.join(missing) or '无'}。"
        + ("配置未登记最终任务清单，当前仅为部分数据报告。" if not expected else ""), ""]
    if frame.empty:
        sections.append("目前尚无策略结果，所有图表等待实际记录，不生成示意实验数据。")
        return finish(run_dir, manifest, sections)

    main_rows = frame[frame.experiment.eq("main")].copy()
    fit = main_rows[main_rows.feasible & main_rows.get("utility", pd.Series(np.nan, index=main_rows.index)).notna()].copy()
    coverage_keys = [c for c in ["experiment", "model", "policy_id", "robust"] if c in frame]
    coverage = frame.groupby(coverage_keys, dropna=False).agg(
        registered=("feasible", "size"), feasible=("feasible", "sum")).reset_index()
    coverage["infeasible"] = coverage.registered - coverage.feasible
    coverage.to_csv(run_dir / "protected_coverage.csv", index=False, encoding="utf-8-sig")
    strict = main_rows[main_rows.policy_id.eq("minimax_six")]
    strict_robust = main_rows[main_rows.policy_id.eq("minimax_six_robust")]
    if not strict.empty:
        finding = (f"预设的严格六子群保护主方法，在 {len(strict)} 个完整数据模型/种子任务中只有 "
                   f"{int(strict.feasible.sum())} 个验证可行。")
        if not strict_robust.empty:
            finding += (f"双折约束变体为 {int(strict_robust.feasible.sum())}/{len(strict_robust)} 可行。")
        finding += ("因此，‘同时满足全部严格要求’尚未普遍实现；不能只展示少数可行种子。"
                    "这些冲突为后续计算最低必要保护松弛提供了直接依据。")
        sections.extend([finding, ""])
    sections.extend([table(coverage, 0), "", "不可行记录全部保留；可行记录的均值不能替代完整任务的通过率。", ""])
    pairs = same_lp_pairs(main_rows)
    pairs.to_csv(run_dir / "protected_matched_lp_baselines.csv", index=False, encoding="utf-8-sig")
    primary_beta = float(config.get("primary_beta", .01))
    primary = primary_scenario(fit, config)
    primary_pairs = primary_scenario(pairs, config)
    if not primary_pairs.empty:
        cols = [c for c in ["utility", "test_profit_retention", "approval_rate", "risk", "abs_frr_gap",
                           "thin_frr", "thick_frr", "worst_subgroup_frr", "utility_change_vs_lp",
                           "thin_frr_change_pp", "thick_frr_change_pp", "worst_subgroup_frr_change_pp",
                           "minimum_required_delta", "effective_delta"] if c in primary_pairs]
        summary = primary_pairs.groupby(["model", "method_role", "robust"], dropna=False)[cols].mean().reset_index()
        n = primary_pairs.groupby(["model", "method_role", "robust"], dropna=False).size().reset_index(name="feasible_rows")
        summary = summary.merge(n, on=["model", "method_role", "robust"], validate="one_to_one")
        summary.to_csv(run_dir / "protected_primary_summary.csv", index=False, encoding="utf-8-sig")
        sections.extend(["## 主设置的同空间业务对照", "", table(summary), "",
            "收益保持率使用测试重放的收益除以同模型 LP 基准的测试收益；收益预算约束是在验证集施加的。"
            "测试保持率高于 100% 或低于验证预算边界都作为观察结果呈现。只有正基准收益才计算比例。", ""])

    # 1. Every prespecified profit budget: no test-dependent policy selection.
    sweep = primary_scenario(pairs, config, include_beta=False)
    if not sweep.empty:
        sweep = sweep[sweep.method_role.str.startswith(("minimax_", "adaptive_")) & sweep.beta.notna()].copy()
    if not sweep.empty and "worst_subgroup_frr" in sweep:
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
        keys = ["model", "method_role", "robust"]
        for index, (combo, subset) in enumerate(sweep.groupby(keys, sort=False)):
            means = subset.groupby("beta")[["test_profit_retention", "worst_subgroup_frr"]].mean().sort_index()
            label = f"{combo[0]} / {ROLE_LABELS.get(combo[1], combo[1])}"
            for ax, metric in zip(axes, ["test_profit_retention", "worst_subgroup_frr"]):
                ax.plot(means.index * 100, means[metric] * 100, "o-", ms=3.5,
                        color=COLORS[index % len(COLORS)], label=label)
        axes[0].axhline(100, ls="--", lw=.8, color="#555555")
        axes[0].set(xlabel="Validation profit-loss allowance (%)", ylabel="Test profit retained vs same LP baseline (%)")
        axes[1].set(xlabel="Validation profit-loss allowance (%)", ylabel="Worst of six subgroup FRRs on test (%)")
        axes[1].legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.suptitle("Frozen profit budgets: business retention and worst-subgroup opportunity")
        fig.tight_layout()
        save(fig, "01_budget_and_worst_group", sweep,
             "Means across all available seeds of each fixed method/budget; incomplete or infeasible jobs remain in coverage tables.")
    else:
        skip("01_budget_and_worst_group", "No completed minimax profit-budget sweep with subgroup outcomes.")

    # 2. All decision families and feasibility; extending policy space is explicit.
    if not main_rows.empty:
        display = primary_scenario(main_rows, config)
        counts = display.groupby(["model", "method_role", "decision_family"], dropna=False).agg(
            feasible=("feasible", "sum"), registered=("feasible", "size")).reset_index()
        counts["rate"] = counts.feasible / counts.registered
        fig, ax = plt.subplots(figsize=(11.8, max(4, .3 * len(counts))))
        y = np.arange(len(counts))
        colors = ["#D38A44" if str(v).startswith("deterministic") else "#277D80" for v in counts.decision_family]
        ax.barh(y, counts.rate * 100, color=colors)
        ax.set_yticks(y, [f"{r.model} / {r.method_role}" for r in counts.itertuples()])
        for yy, r in zip(y, counts.itertuples()):
            ax.text(min(101, r.rate * 100 + 1), yy, f"{int(r.feasible)}/{int(r.registered)}", va="center", fontsize=7)
        ax.invert_yaxis()
        ax.set(xlim=(0, 117), xlabel="Validation-feasible registered policies (%)",
               title="Feasibility: deterministic hard thresholds and LP mixtures are different policy spaces")
        fig.tight_layout()
        save(fig, "02_policy_space_feasibility", display,
             "Registered finite primary-budget rules, including infeasible cases. A larger feasible set is an optimization-space change.")
        sections.extend(["## 决策空间与可行性", "", table(counts.drop(columns="rate"), 0), "",
                         "可行性提升需要同时查看决策空间变化与约束差异；不能仅凭 LP 更易可行就声称算法全面优于硬阈值。", ""])

    # 3. Six-group changes relative to the same LP baseline at the primary budget.
    changes = [c + "_change_pp" for c in SUBGROUPS if c + "_change_pp" in primary_pairs]
    heat = primary_pairs[~primary_pairs.method_role.eq("lp_profit_baseline")].copy() if not primary_pairs.empty else primary_pairs
    if not heat.empty and len(changes) == 6:
        means = heat.groupby(["model", "method_role", "robust"], sort=False)[changes].mean()
        fig, ax = plt.subplots(figsize=(10, max(4, .39 * len(means))))
        lim = max(.1, float(np.nanmax(np.abs(means.to_numpy()))))
        im = ax.imshow(means.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(np.arange(6), SUBGROUP_LABELS, rotation=25, ha="right")
        counts = heat.groupby(["model", "method_role", "robust"], sort=False).size()
        ax.set_yticks(np.arange(len(means)), [f"{a} / {b} (n={int(counts.loc[(a,b,c)])})" for a, b, c in means.index])
        for i in range(len(means)):
            for j in range(6):
                ax.text(j, i, f"{means.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.grid(False)
        ax.set_title("Six-group FRR changes vs same LP profit baseline (pp; negative = fewer normal applicants rejected)")
        fig.colorbar(im, ax=ax, shrink=.65, label="FRR change (percentage points)")
        fig.tight_layout()
        save(fig, "03_six_group_changes", heat, "Fixed primary budget; six groups are credit-history group crossed with training-derived income terciles.")
        sections.extend(["## 六个子群的机会变化", "", table(means.reset_index()), "",
                         "负值表示相对同空间收益基准少拒绝正常标签客户；正值表示增加。收入分组只用于目标与审计，"
                         "审批概率仍来自信用记录组内单调的分数阈值混合。", ""])
    else:
        skip("03_six_group_changes", "Six subgroup outcomes and matched LP baselines not yet available.")

    # 4. Holdout violations are observations, never a reason to retune a policy.
    violations = constraint_rates(primary)
    violations.to_csv(run_dir / "protected_test_constraints.csv", index=False, encoding="utf-8-sig")
    if not violations.empty:
        compare_roles = (["adaptive_primary", "adaptive_robust"] if
                         violations.method_role.str.startswith("adaptive_").any() else
                         ["minimax_six", "minimax_six_robust"])
        matched_variants = violations[violations.method_role.isin(compare_roles)]
        summary = matched_variants.groupby(["robust", "constraint"])[["violations", "evaluated"]].sum().reset_index()
        if summary.empty:
            summary = violations.groupby(["robust", "constraint"])[["violations", "evaluated"]].sum().reset_index()
        summary["rate"] = summary.violations / summary.evaluated
        labels = list(dict.fromkeys(summary.constraint))
        fig, ax = plt.subplots(figsize=(9.3, 4.5))
        x = np.arange(len(labels))
        for offset, robust in enumerate(sorted(summary.robust.unique())):
            subset = summary[summary.robust.eq(robust)].set_index("constraint").reindex(labels)
            ax.bar(x + (offset - .5) * .33, subset.rate * 100, width=.31,
                   color=COLORS[offset], label="Two-fold constraints" if robust else "Nominal constraints")
        ax.set_xticks(x, labels)
        ax.set(ylabel="Observed violation rate among applicable test checks (%)",
               title="Validation constraints may transfer imperfectly to test outcomes")
        ax.legend()
        fig.tight_layout()
        save(fig, "04_test_constraint_transfer", primary,
             "Descriptive checks of nominal and two-fold six-group minimax at the fixed primary budget, where available. Infeasible cases are tabulated separately.")
        sections.extend(["## 验证约束在测试集的迁移", "", table(violations, 4), "",
                         "每项只统计可行策略中适用且有记录的检查。名义与双折约束的任务构成可能不同，"
                         "整体比例只作描述；应结合逐模型、逐方法的分母和不可行数量判断。", ""])
    else:
        skip("04_test_constraint_transfer", "No recorded applicable test-constraint flags.")

    # 5. Same decision-space outcomes. A connected validation budget path is not a
    # test-selected Pareto frontier; all points are retained, including dominated ones.
    lp = pairs[pairs.decision_family.eq("group_threshold_mixture")].copy() if not pairs.empty else pairs
    if not lp.empty and {"worst_subgroup_frr", "test_profit_retention"}.issubset(lp):
        names = list(dict.fromkeys(lp.model))
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.7), squeeze=False)
        roles = list(dict.fromkeys(lp.method_role))
        markers = ["D", "o", "s", "^", "P", "X", "v"]
        for ax, model in zip(axes[0], names):
            for i, role in enumerate(roles):
                subset = lp[lp.model.eq(model) & lp.method_role.eq(role)]
                ax.scatter(subset.worst_subgroup_frr * 100, subset.test_profit_retention * 100,
                           s=30, alpha=.65, color=COLORS[i % len(COLORS)], marker=markers[i % len(markers)],
                           label=ROLE_LABELS.get(role, role))
            ax.axhline(100, ls="--", lw=.8, color="#666666")
            ax.set(xlabel="Worst six-group test FRR (%)", ylabel="Test profit retained vs same LP baseline (%)", title=model)
        axes[0, -1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle("All frozen LP policies: profit and subgroup opportunity trade-offs")
        fig.tight_layout()
        save(fig, "05_same_space_tradeoffs", lp,
             "Every saved feasible LP point, including dominated points; no best-on-test model or budget selection.")
    else:
        skip("05_same_space_tradeoffs", "Matched LP outcomes not yet available.")

    # 6. Adaptive protection explicitly reports the protection relaxation that
    # validation feasibility requires, rather than silently claiming 1pp holds.
    adaptive = primary_pairs[primary_pairs.policy_id.str.startswith("adaptive_")].copy() if not primary_pairs.empty else primary_pairs
    if not adaptive.empty and "minimum_required_delta" in adaptive:
        adaptive_summary_cols = [c for c in ["minimum_required_delta", "effective_delta", "requested_delta",
                                            "test_profit_retention", "worst_subgroup_frr",
                                            "worst_subgroup_frr_change_pp", "thin_frr_change_pp",
                                            "thick_frr_change_pp"] if c in adaptive]
        adaptation = adaptive.groupby(["model", "method_role", "robust"], sort=False)[adaptive_summary_cols].mean().reset_index()
        adaptation.to_csv(run_dir / "protected_adaptive_summary.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(11, 4.6))
        y = np.arange(len(adaptation))
        required = adaptation.minimum_required_delta * 100
        allowed = adaptation.effective_delta * 100 if "effective_delta" in adaptation else required
        ax.barh(y, allowed, color="#ABC9CA", label="Effective protection cap incl. fixed slack")
        ax.barh(y, required, color="#247B7C", label="Minimum required cap on validation")
        first_dots = True
        for yy, item in enumerate(adaptation.itertuples()):
            values = adaptive[adaptive.model.eq(item.model) & adaptive.method_role.eq(item.method_role)
                              & adaptive.robust.eq(item.robust)].minimum_required_delta.dropna().to_numpy() * 100
            offsets = np.linspace(-.11, .11, len(values)) if len(values) > 1 else np.zeros(len(values))
            ax.scatter(values, yy + offsets, color="#252525", s=23, zorder=5,
                       label="Individual validation seeds" if first_dots else None)
            first_dots = False
        target = float(config.get("primary_delta", .01)) * 100
        ax.axvline(target, color="#BB5B3D", ls="--", label=f"Original requested protection ({target:g} pp)")
        ax.set_yticks(y, [f"{r.model} / {r.method_role}" for r in adaptation.itertuples()])
        ax.invert_yaxis()
        ax.set(xlabel="Maximum subgroup FRR deterioration allowance (percentage points)",
               title="Minimum required protection relaxation: bars = seed means, dots = individual seeds")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        save(fig, "06_minimum_required_harm", adaptive,
             "Primary fixed 1% profit budget. Validation first minimizes the necessary worst-subgroup deterioration; a prespecified 0.1pp slack is then added before worst-FRR/profit optimization.")
        above = adaptive.minimum_required_delta > adaptive.get("requested_delta", config.get("primary_delta", .01)) + 1e-7
        sections.extend(["## 面对不可行约束的最低必要损害方案", "",
            "后续有限扩展采用两阶段设计：先在原风险、资金、公平差距和收益下限下，计算最低必要的"
            "最大子群误拒率恶化幅度 d*；然后将保护上限冻结为 d* 加预设 0.1 个百分点，"
            "在这个明确的保护范围内最小化六子群最差误拒率，再以收益作平局选择。"
            "所有数值来自验证集，测试结果不会改变保护上限。", "",
            f"在主设置的 {len(adaptive)} 个可行自适应任务中，有 {int(above.sum())} 个的最低必要保护容差已超过原先请求的 1 个百分点。"
            "因此该方案提供的是可量化、尽量小的保护松弛，不能描述为严格 1pp 保护已经实现。", "",
            table(adaptation), "",
            "先求最低损害再改善最差子群，是为了把不能同时满足的要求变成可解释的业务选择。"
            "业务需要共同查看 d*、收益保持率、六组机会变化和测试超限，不能只看总公平差距。", ""])
    else:
        skip("06_minimum_required_harm", "No adaptive minimum-harm outcomes in this run; strict infeasibility remains visible.")

    standard = load_csv(run_dir / "standard_method_comparison.csv")
    if not standard.empty and {"model", "method", "utility", "worst_subgroup_frr"}.issubset(standard):
        standard["utility_per_1000_applicants"] = standard.utility * 1000 / standard.records
        names = list(dict.fromkeys(standard.model))
        plot_methods = ["lp_profit_baseline", "cost_sensitive_fairlearn", "det_gap", "lp_gap", "adaptive_primary"]
        plot_standard = standard[standard.method.isin(plot_methods)].copy()
        methods = [m for m in plot_methods if m in set(plot_standard.method)]
        fig, axes = plt.subplots(1, len(names), figsize=(6.1 * len(names), 4.8), squeeze=False)
        markers = ["D", "o", "s", "^", "P", "X", "v"]
        for ax, model in zip(axes[0], names):
            for i, method in enumerate(methods):
                subset = plot_standard[plot_standard.model.eq(model) & plot_standard.method.eq(method)]
                if subset.empty:
                    continue
                ax.scatter(subset.worst_subgroup_frr * 100, subset.utility_per_1000_applicants,
                           marker=markers[i % len(markers)], color=COLORS[i % len(COLORS)],
                           s=29, alpha=.4)
                ax.scatter(subset.worst_subgroup_frr.mean() * 100, subset.utility_per_1000_applicants.mean(),
                           marker=markers[i % len(markers)], color=COLORS[i % len(COLORS)],
                           s=76, edgecolors="white", linewidth=.7, label=method)
            ax.set(xlabel="Worst six-group test FRR (%)", ylabel="Scenario contribution / 1,000 applicants", title=model)
        axes[0, -1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle("Standard methods and our fixed primary method: every completed seed plus means")
        fig.tight_layout()
        save(fig, "07_standard_method_comparison", plot_standard,
             "Existing validation-frozen cost-sensitive Fairlearn at tolerance 0.02; LP and deterministic rules retain their own decision spaces. Large markers are seed means, small markers are all seeds.")
        metrics = [c for c in ["utility", "approval_rate", "risk", "abs_frr_gap", "worst_subgroup_frr",
                              "thin_frr", "thick_frr"] if c in standard]
        standard_means = standard.groupby(["model", "method"], sort=False)[metrics].mean().reset_index()
        sections.extend(["## 与标准公平方法和原硬阈值的实际比较", "", table(standard_means), "",
            "Fairlearn 使用已保存的成本敏感、验证选择版本，容差固定为 2 个百分点。本比较展示已有冻结方法的真实测试结果；"
            "不同方法的优化目标、随机化方式与网格不同，不能据此直接宣称全面支配或论文 SOTA。", ""])
        for model in names:
            subset = standard_means[standard_means.model.eq(model)].set_index("method")
            if {"adaptive_primary", "cost_sensitive_fairlearn"}.issubset(subset.index):
                own, ref = subset.loc["adaptive_primary"], subset.loc["cost_sensitive_fairlearn"]
                text = (f"{model}：自适应主方法相对成本敏感 Fairlearn 的平均单位情景净贡献变化 "
                        f"{own.utility-ref.utility:+.2f}，最差子群误拒率变化 "
                        f"{(own.worst_subgroup_frr-ref.worst_subgroup_frr)*100:+.3f} 个百分点。")
                if "det_gap" in subset.index:
                    hard = subset.loc["det_gap"]
                    text += (f"相对原硬阈值公平规则，净贡献变化 {own.utility-hard.utility:+.2f}，"
                             f"最差子群误拒率变化 {(own.worst_subgroup_frr-hard.worst_subgroup_frr)*100:+.3f} 个百分点，"
                             f"对照组误拒率变化 {(own.thick_frr-hard.thick_frr)*100:+.3f} 个百分点。")
                sections.extend([text, ""])
        sections.extend(["这些比较共同说明收益、最差子群与附带误拒损害之间存在权衡；方法的价值应按明确目标判断。", ""])
    else:
        skip("07_standard_method_comparison", "Standard comparison CSV is not available in this run.")

    intervals = load_csv(run_dir / "paired_intervals.csv")
    standard_intervals = load_csv(run_dir / "standard_method_paired_intervals.csv")
    replays = load_csv(run_dir / "randomization_replays.csv")
    if not intervals.empty or not standard_intervals.empty or not replays.empty:
        sections.extend(["## 区间和随机执行的不确定性", "",
            "`paired_intervals.csv` 中的区间是固定模型、固定审批规则下的申请人配对自助区间，"
            "反映样本重抽的不确定性；不能替代训练随机种子比较，也不能代表重新调参后的整体不确定性。"
            "`randomization_replays.csv` 另外反映逐申请人独立随机获批的一次实际执行波动。", ""])
        if not standard_intervals.empty:
            sections.extend(["与标准 Fairlearn 的配对区间保存在 `standard_method_paired_intervals.csv`。"
                             "按预设方法、逐种子呈现，不按区间是否显著来筛选种子。", ""])
        if not replays.empty:
            replay_summary = replays.groupby(["job", "policy"], sort=False).agg(
                repetitions=("repetition", "size"), utility_mean=("utility", "mean"),
                utility_sd=("utility", "std"), risk_mean=("risk", "mean"), risk_sd=("risk", "std"),
                risk_met_rate=("risk_met", "mean"), gap_met_rate=("gap_met", "mean"),
                profit_floor_met_rate=("profit_floor_met", "mean")).reset_index()
            replay_summary.to_csv(run_dir / "protected_randomization_summary.csv", index=False, encoding="utf-8-sig")
            sections.extend([table(replay_summary), "",
                             "满足比例来自保存的有限随机执行次数，只描述这些执行；它不是银行未来每一批贷款满足约束的保证。", ""])

    stress = frame[frame.experiment.eq("stress")]
    if not stress.empty:
        stress_keys = [c for c in ["model", "sampling_condition", "policy_id", "robust"] if c in stress]
        stress_coverage = stress.groupby(stress_keys, dropna=False).agg(
            registered=("feasible", "size"), feasible=("feasible", "sum")).reset_index()
        sections.extend(["## 偏差压力测试覆盖", "", table(stress_coverage, 0), "",
                         "压力任务使用既定偏置数据训练得到的保存分数；本扩展只重做冻结审批优化。"
                         "各偏差条件应配对同条件的 LP 基准，不能与完整数据基准混算收益成本。", ""])
        stress_pairs = same_lp_pairs(stress)
        stress_pairs.to_csv(run_dir / "protected_stress_matched.csv", index=False, encoding="utf-8-sig")
    audit_path = run_dir / "PLAN_COMPLETION_AUDIT_CN.md"
    if audit_path.exists():
        sections.extend(["## 原计划逐项核对", "", f"完整核对见 [{audit_path.name}]({audit_path.name})。", ""])
    sections.extend(["## 方法结论应怎样写", "",
        "本轮回答的是：在固定风险、资金与可容忍收益损失下，能否降低最不利子群的正常客户误拒率，"
        "以及代价由谁承担。六子群 minimax、收益下限和相对基准保护共同构成可解释的审批设计。", "",
        "阈值混合、线性规划和 minimax 都是已有思想；课程贡献是把它们与当前 Home Credit 的"
        "偏差机制、透明收益情景和业务预算结合，并做同空间消融与测试超限审计。"
        "若某些保护条件不可行、测试收益低于预算或双折约束未改善迁移，保留结果并说明适用条件。", "",
        "## 输出文件", "",
        "- `policy_metrics.csv`：所有登记策略，包括不可行任务。",
        "- `protected_matched_lp_baselines.csv`：同空间收益和机会变化的配对明细。",
        "- `protected_test_constraints.csv`：逐方法测试约束超限次数与分母。",
        "- `figures/protected_*.png/.pdf`：可用于汇报的真实结果图。",
        "- `figure_data/protected_*.csv`：每张图的可追溯源记录。",
        "- `protected_figure_manifest.json`：输入、脚本、配置和输出校验值。", ""])
    for item in manifest["figures"]:
        sections.extend([f"![{item['caption']}](figures/protected_{item['name']}.png)", ""])
    return finish(run_dir, manifest, sections)


def finish(run_dir, manifest, sections):
    (run_dir / "METHODS_RESULTS_CN.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    (run_dir / "protected_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = main(args.run, require_complete=args.require_complete)
    print(json.dumps({"figures": len(result["figures"]), "skipped": result["skipped"],
                      "missing_jobs": result["missing_jobs"]}, ensure_ascii=False))
