"""Regenerate presentation figures and a Chinese interpretation from saved runs.

This module never fits models, selects thresholds, or changes experimental data.
Every plotted point retains its source row in figure_data/*.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
COLORS = ["#216869", "#D97732", "#4263A0", "#9163A4", "#739D46", "#C65865"]
LABELS = {
    "common_profit": "Common profit threshold",
    "group_max_approval": "Maximum approvals",
    "group_profit": "Group profit thresholds",
    "group_profit_gap": "Profit + fairness",
    "group_profit_gap_noharm": "Profit + fairness + protection",
    "fairlearn_business": "Fairlearn, cost/risk tuned",
}
ALIASES = {"approvals": "approval_count", "difficulty_count": "bad_approved",
           "risk": "approved_risk", "approved_amount": "total_principal",
           "amount_utility": "amount_profit", "noharm_delta": "delta",
           "utility_per_1000_applicants": "utility_per_1000"}


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["source_file"] = str(path.resolve())
    frame["source_row"] = np.arange(len(frame)) + 2
    return frame


def normalize(frame):
    if frame.empty:
        return frame
    frame = frame.copy()
    for source, target in ALIASES.items():
        if target not in frame and source in frame:
            frame[target] = frame[source]
    if "family" not in frame:
        frame["family"] = frame.get("policy_type", pd.Series("unknown", index=frame.index))
    if "utility_per_1000" not in frame and "utility" in frame:
        if "n_test" in frame:
            size = frame.n_test
        else:
            size = frame.approval_count / frame.approval_rate.replace(0, np.nan)
        frame["utility_per_1000"] = frame.utility * 1000 / size
    for column in ["risk_budget", "loss_ratio", "epsilon", "delta", "capital_fraction"]:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    parts = [column for column in ["model", "seed", "policy_id", "scenario", "loss_ratio",
                                   "risk_budget", "epsilon", "delta", "capital_fraction"] if column in frame]
    frame["point_id"] = frame[parts].astype(str).agg("|".join, axis=1)
    return frame


def near(frame, column, value):
    return np.isclose(pd.to_numeric(frame[column], errors="coerce"), value, atol=1e-10)


def unit_rows(frame):
    if "utility_basis" in frame:
        return frame[frame.utility_basis.eq("unit_utility") | frame.family.eq("fairlearn")]
    if "objective" not in frame:
        return frame
    return frame[~frame.objective.fillna("").str.contains("amount", case=False)]


def primary(frame, config, loss=True):
    if frame.empty:
        return frame
    mask = near(frame, "risk_budget", config.get("primary_risk_budget", .05))
    mask &= near(frame, "capital_fraction", 1.) | frame.capital_fraction.isna()
    if loss:
        mask &= near(frame, "loss_ratio", config.get("primary_loss_ratio", 5))
    return unit_rows(frame[mask])


def completed(frame):
    if frame.empty:
        return frame
    if "utility" not in frame:
        return frame.iloc[:0]
    mask = pd.Series(True, index=frame.index)
    if "status" in frame:
        mask &= frame.status.astype(str).str.upper().isin(["PASS", "OK", "FEASIBLE", "SUCCESS"])
    if "feasible" in frame:
        mask &= ~frame.feasible.astype(str).str.lower().isin(["false", "0"])
    return frame[mask & frame.utility.notna()]


def baseline_pairs(frame, config, all_losses=False):
    """Match within model/seed/loss/risk/capital/objective, without test selection."""
    frame = primary(completed(frame), config, loss=not all_losses)
    fair = frame[(frame.family == "group_profit_gap") & near(frame, "epsilon", config.get("primary_epsilon", .02))]
    base = frame[frame.family == "group_profit"]
    keys = [key for key in ["model", "seed", "loss_ratio", "risk_budget", "capital_fraction", "objective"] if key in frame]
    if fair.empty or base.empty:
        return pd.DataFrame()
    return fair.merge(base, on=keys, suffixes=("_fair", "_base"), validate="many_to_one")


def table_md(frame):
    if frame.empty:
        return "（尚无完成记录）"
    clean = frame.fillna("NA")
    rows = ["| " + " | ".join(map(str, clean.columns)) + " |",
            "| " + " | ".join(["---"] * len(clean.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in clean.itertuples(index=False, name=None))
    return "\n".join(rows)


def main(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(run_dir / "config_frozen.json") or read_json(run_dir / "config.json") or read_json(HERE / "config.json", {})
    models = load_csv(run_dir / "model_metrics.csv")
    main_policies = load_csv(run_dir / "policy_metrics.csv")
    business_policies = load_csv(run_dir / "fairlearn_business" / "metrics.csv")
    policies = normalize(pd.concat([frame for frame in [main_policies, business_policies] if not frame.empty], ignore_index=True)
                         if not main_policies.empty or not business_policies.empty else pd.DataFrame())
    status = read_json(run_dir / "status.json", {})
    figures = run_dir / "figures"
    sources = run_dir / "figure_data"
    figures.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    manifest = {"run_directory": str(run_dir), "selection": "Configuration selected without test optimization; saved outcomes only.",
                "config_sha256": digest(run_dir / "config_frozen.json") if (run_dir / "config_frozen.json").exists() else None,
                "figures": [], "skipped": []}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "savefig.dpi": 300, "pdf.fonttype": 42})

    def save(fig, name, source_frame, description):
        png, pdf, csv = figures / f"{name}.png", figures / f"{name}.pdf", sources / f"{name}.csv"
        source_frame.to_csv(csv, index=False, encoding="utf-8-sig")
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        manifest["figures"].append({"name": name, "description": description, "source_rows": len(source_frame),
                                    "files": {str(path.relative_to(run_dir)): digest(path) for path in [png, pdf, csv]}})

    def skip(name, reason):
        manifest["skipped"].append({"name": name, "reason": reason})

    # Figure 1: model discrimination and measured training time, all saved seeds.
    if not models.empty and {"model", "auc", "train_seconds"}.issubset(models):
        summary = models.groupby("model", sort=False).agg(auc=("auc", "mean"), sd=("auc", "std"),
                                                        seconds=("train_seconds", "mean"), seeds=("auc", "count"))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        x = np.arange(len(summary))
        model_colors = [COLORS[index % len(COLORS)] for index in range(len(summary))]
        axes[0].bar(x, summary.auc, color=model_colors, yerr=summary.sd.fillna(0), capsize=3)
        lower = max(0., float(summary.auc.min()) - .025)
        axes[0].set(ylim=(lower, min(1., float(summary.auc.max()) + .025)), ylabel="Test ROC AUC", title="Predictive discrimination")
        axes[1].bar(x, summary.seconds.clip(lower=.001), color=model_colors)
        axes[1].set(yscale="log", ylabel="Measured training seconds (log scale)", title="Training cost")
        for ax in axes:
            ax.set_xticks(x, [f"{name}\n(n={int(n)})" for name, n in zip(summary.index, summary.seeds)])
        fig.suptitle("Same feature set and split; completed training seeds only")
        fig.tight_layout()
        save(fig, "01_models_accuracy_and_time", models, "Mean test AUC and measured training time; bars are SD across completed seeds.")
        metrics_to_summarize = [key for key in ["auc", "brier", "average_precision", "train_seconds", "reference_auc", "n_test", "input_features"] if key in models]
        comparison = models.groupby("model", sort=False)[metrics_to_summarize].agg(["mean", "std", "count"])
        comparison.columns = ["_".join(column) for column in comparison.columns]
        comparison.reset_index().to_csv(run_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    else:
        skip("01_models_accuracy_and_time", "Model metrics not yet available.")

    if policies.empty:
        write_json(run_dir / "figure_manifest.json", manifest)
        note = "当前运行尚未写出策略结果。报告脚本已就绪，完成任一训练任务后可重新运行，不需要重新训练。\n"
        (run_dir / "RESULTS_CN.txt").write_text(note, encoding="utf-8")
        (run_dir / "RESULTS_CN.md").write_text(note, encoding="utf-8")
        return manifest

    fit = completed(policies)
    prim = primary(fit, config)
    names = list(dict.fromkeys(policies.model.astype(str)))
    color = {name: COLORS[index % len(COLORS)] for index, name in enumerate(names)}

    # Figure 2: test outcomes of every frozen primary-scenario policy, not test-selected winners.
    if not prim.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        markers = {"common_profit": "s", "group_max_approval": "v", "group_profit": "D", "group_profit_gap": "o", "group_profit_gap_noharm": "^", "fairlearn_business": "P"}
        for (model, family), frame in prim.groupby(["model", "family"], sort=False):
            ax.scatter(frame.abs_frr_gap * 100, frame.utility_per_1000, s=28, alpha=.62,
                       color=color[str(model)], marker=markers.get(family, "x"))
        first = ax.legend(handles=[Line2D([], [], marker="o", ls="", color=color[name], label=name) for name in names],
                          title="Prediction backend", loc="lower left", bbox_to_anchor=(1.01, 0), fontsize=8)
        ax.add_artist(first)
        seen = list(dict.fromkeys(prim.family))
        ax.legend(handles=[Line2D([], [], marker=markers.get(name, "x"), ls="", color="#444444", label=LABELS.get(name, name)) for name in seen],
                  title="Frozen policy", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
        ax.set(xlabel="Test absolute FRR gap (percentage points)", ylabel="Scenario net contribution / 1,000 applicants",
               title=f"Fairness and business outcomes: loss ratio {config.get('primary_loss_ratio', 5)}, risk reference {config.get('primary_risk_budget', .05):.0%}")
        fig.tight_layout()
        save(fig, "02_fairness_business_frontier", prim, "All completed primary-scenario frozen policies. Threshold policies enforce the validation risk budget; standard Fairlearn does not impose that business constraint.")
    else:
        skip("02_fairness_business_frontier", "No feasible primary-scenario policies.")

    # Figure 3: fairness strength; identical policy family and scenario.
    sweep = prim[prim.family.eq("group_profit_gap")].copy()
    if not sweep.empty:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
        for model, frame in sweep.groupby("model", sort=False):
            means = frame.groupby("epsilon")[["approval_rate", "approved_risk", "utility_per_1000"]].mean().sort_index()
            for ax, column, multiplier in zip(axes, ["approval_rate", "approved_risk", "utility_per_1000"], [100, 100, 1]):
                ax.plot(means.index * 100, means[column] * multiplier, "o-", ms=4, label=model, color=color[str(model)])
        for ax, ylabel in zip(axes, ["Test approval rate (%)", "Test repayment-difficulty rate (%)", "Net contribution / 1,000 applicants"]):
            ax.set(xlabel="Allowed validation FRR gap (pp)", ylabel=ylabel)
            ax.invert_xaxis()
        axes[1].axhline(config.get("primary_risk_budget", .05) * 100, color="#666666", ls="--", lw=1, label="Validation budget")
        axes[0].legend(fontsize=8)
        fig.suptitle("Tighter validation fairness constraints: observed test business effects")
        fig.tight_layout()
        save(fig, "03_fairness_strength_business", sweep, "Means across available seeds; validation constraints need not hold exactly on test data.")
    else:
        skip("03_fairness_strength_business", "No completed fairness sweep.")

    pairs = baseline_pairs(policies, config)
    # Predeclared backend and all its completed seeds, rather than choosing test profit.
    reference_model = config.get("report_primary_model", "xgboost")
    if not pairs.empty:
        if reference_model not in set(pairs.model):
            reference_model = sorted(set(pairs.model))[0]
        reference = pairs[pairs.model.eq(reference_model)].sort_values("seed")
        one = reference.select_dtypes(include="number").mean()
        point_ids = reference.point_id_base.to_list() + reference.point_id_fair.to_list()
        raw = policies[policies.point_id.isin(point_ids)].copy()
        seed_caption = f"mean of {len(reference)} completed seed(s)"
        loss = float(one.loss_ratio)
        components = np.array([one.good_approved_fair - one.good_approved_base,
                               -loss * (one.bad_approved_fair - one.bad_approved_base)])
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(0, components[0], color=COLORS[0] if components[0] >= 0 else COLORS[1])
        ax.bar(1, components[1], bottom=components[0], color=COLORS[0] if components[1] >= 0 else COLORS[1])
        ax.bar(2, components.sum(), color=COLORS[2])
        for index, value in enumerate([*components, components.sum()]):
            top = components[0] + components[1] if index == 1 else value
            ax.annotate(f"{value:+,.1f}", (index, top), xytext=(0, 7 if value >= 0 else -15), textcoords="offset points", ha="center")
        low = min(0., components[0], components.sum())
        high = max(0., components[0], components.sum())
        span = max(1., high - low)
        ax.set_ylim(low - .20 * span, high + .20 * span)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set_xticks([0, 1, 2], ["Change from normal\nloan approvals", "Change from difficulty\nloan approvals", "Total contribution\nchange"])
        if np.allclose(components, 0):
            ax.set_ylim(-1, 1)
            ax.text(.5, .75, "Approved-label counts and contribution unchanged", transform=ax.transAxes, ha="center", fontsize=10)
        ax.set(ylabel="Scenario net contribution change", title=f"Business effect decomposition: {reference_model}, {seed_caption}")
        fig.tight_layout()
        save(fig, "04_business_contribution_waterfall", raw, "Fair group-profit policy minus unconstrained group-profit policy, same backend/seed/scenario.")
        pd.DataFrame({"component": ["normal_loan_change", "difficulty_loan_change", "total_change"],
                      "contribution": [*components, components.sum()],
                      "source_base": json.dumps(reference.point_id_base.to_list()),
                      "source_fair": json.dumps(reference.point_id_fair.to_list())}).to_csv(sources / "04_waterfall_components.csv", index=False)

        # Figure 6 explicitly exposes opportunity redistribution.
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(2)
        old = np.array([one.thin_frr_base, one.thick_frr_base]) * 100
        new = np.array([one.thin_frr_fair, one.thick_frr_fair]) * 100
        ax.bar(x - .18, old, .36, label="Group profit baseline", color=COLORS[2])
        ax.bar(x + .18, new, .36, label="Profit + fairness", color=COLORS[0])
        for index, difference in enumerate(new - old):
            ax.text(index, max(old[index], new[index]) + .6, f"{difference:+.2f} pp", ha="center")
        ax.set_xticks(x, ["Thin credit history", "Comparison group"])
        ax.set(ylim=(0, max(old.max(), new.max()) + 5), ylabel="Test normal-label rejection rate (%)",
               title=f"Group opportunity changes: {reference_model}, {seed_caption}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save(fig, "06_group_opportunity_changes", raw, "Groupwise FRR changes: reduced gap can include increased rejection in another group.")
    else:
        skip("04_business_contribution_waterfall", "No matched baseline/fair policy at primary epsilon.")
        skip("06_group_opportunity_changes", "No matched baseline/fair policy at primary epsilon.")

    sensitivity = baseline_pairs(policies, config, all_losses=True)
    if not sensitivity.empty:
        sensitivity["delta_utility_per_1000"] = sensitivity.utility_per_1000_fair - sensitivity.utility_per_1000_base
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for model, frame in sensitivity.groupby("model", sort=False):
            mean = frame.groupby("loss_ratio")[["delta_utility_per_1000", "abs_frr_gap_fair"]].mean()
            axes[0].plot(mean.index, mean.delta_utility_per_1000, "o-", label=model, color=color[str(model)])
            axes[1].plot(mean.index, mean.abs_frr_gap_fair * 100, "o-", label=model, color=color[str(model)])
        axes[0].axhline(0, color="#666666", ls="--", lw=1)
        axes[0].set(xlabel="Difficulty-loss / normal-return ratio", ylabel="Fair minus baseline contribution / 1,000", title="Business cost under alternative loss scenarios")
        axes[1].set(xlabel="Difficulty-loss / normal-return ratio", ylabel="Test absolute FRR gap (pp)", title="Fairness under the same scenarios")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        save(fig, "05_loss_scenario_sensitivity", sensitivity, "Matched same-model baselines per loss scenario; values are not compared across currencies.")
    else:
        skip("05_loss_scenario_sensitivity", "No matched loss-scenario sweep.")

    # Figure 7 uses the separate amount objective and common applicant capital base.
    amount = fit[fit.get("utility_basis", pd.Series("", index=fit.index)).eq("amount_utility")].copy()
    if not amount.empty:
        amount = amount[near(amount, "risk_budget", config.get("primary_risk_budget", .05)) &
                        near(amount, "loss_ratio", config.get("primary_loss_ratio", 5))]
        amount = amount[amount.family.eq("group_profit") |
                        (amount.family.eq("group_profit_gap") & near(amount, "epsilon", config.get("primary_epsilon", .02)))]
    if not amount.empty:
        capitals = sorted(amount.capital_fraction.unique(), reverse=True)
        fig, axes = plt.subplots(1, len(capitals), figsize=(6.2 * len(capitals), 4.8), squeeze=False)
        markers = {"group_profit": "D", "group_profit_gap": "o"}
        for ax, cap in zip(axes[0], capitals):
            for (model, family), frame in amount[near(amount, "capital_fraction", cap)].groupby(["model", "family"], sort=False):
                ax.scatter(frame.abs_frr_gap * 100, frame.amount_profit / 1e6, color=color[str(model)],
                           marker=markers[family], s=40, alpha=.7, label=f"{model}: {LABELS[family]}")
            ax.set(xlabel="Test absolute FRR gap (pp)", ylabel="Scenario amount contribution (million source units)",
                   title=f"Validation funding cap: {cap:.0%} of requested capital")
        axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("Amount-weighted business model: same applicants, same return/loss assumptions")
        fig.tight_layout()
        save(fig, "07_amount_business_and_capital", amount,
             "Amount-profit objective; both capital caps use total requested applicant principal as denominator. Test spending can differ from validation bounds.")
    else:
        skip("07_amount_business_and_capital", "No completed amount-objective policies.")

    business_comparison = prim[prim.family.isin(["group_profit_gap", "fairlearn_business"]) &
                               near(prim, "epsilon", config.get("primary_epsilon", .02))].copy()
    if not business_comparison.empty and business_comparison.family.eq("fairlearn_business").any():
        display_models = [name for name in names if name in set(business_comparison.model)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
        x = np.arange(len(display_models))
        for index, (family, label, c) in enumerate([("group_profit_gap", "Profit + fairness thresholds", COLORS[0]),
                                                   ("fairlearn_business", "Fairlearn: validation cost/risk tuning", COLORS[1])]):
            subset = business_comparison[business_comparison.family.eq(family)]
            grouped = subset.groupby("model")[["utility_per_1000", "abs_frr_gap", "approved_risk"]].agg(["mean", "std"]).reindex(display_models)
            for ax, metric, multiplier in zip(axes, ["utility_per_1000", "abs_frr_gap", "approved_risk"], [1, 100, 100]):
                ax.bar(x + (index - .5) * .36, grouped[(metric, "mean")] * multiplier, width=.36,
                       yerr=grouped[(metric, "std")].fillna(0) * multiplier, capsize=2, label=label, color=c)
        for ax in axes:
            ax.set_xticks(x, display_models, rotation=25, ha="right")
        axes[0].set(ylabel="Net contribution / 1,000 applicants", title="Business objective")
        axes[1].set(ylabel="Test absolute FRR gap (pp)", title="Fairness outcome")
        axes[2].set(ylabel="Difficulty share among approved (%)", title="Risk outcome")
        axes[1].axhline(config.get("primary_epsilon", .02) * 100, color="#666666", ls="--", lw=1)
        axes[2].axhline(config.get("primary_risk_budget", .05) * 100, color="#666666", ls="--", lw=1)
        axes[0].legend(fontsize=7)
        fig.suptitle("Business-aligned baseline: same validation risk budget and fairness tolerance; mean +/- seed SD")
        fig.tight_layout()
        save(fig, "08_business_aligned_fairlearn", business_comparison,
             "Unmodified Fairlearn with validation-only negative-class cost replication and risk-screened cost selection; expected stochastic outcomes compared with deterministic thresholds.")
        business_comparison.to_csv(run_dir / "business_baseline_comparison.csv", index=False, encoding="utf-8-sig")
    else:
        skip("08_business_aligned_fairlearn", "Business-aligned Fairlearn has no completed primary-tolerance policy yet.")

    # Compact factual reporting, with partial completion and infeasibility visible.
    cols = [key for key in ["model", "seed", "policy_id", "family", "objective", "epsilon", "delta", "approval_count", "approval_rate",
                            "approved_risk", "thin_frr", "thick_frr", "abs_frr_gap", "utility", "utility_per_1000", "total_principal", "amount_profit"] if key in prim]
    selected = prim[(prim.family.isin(["common_profit", "group_max_approval", "group_profit"])) |
                    near(prim, "epsilon", config.get("primary_epsilon", .02))].copy()
    selected[cols].to_csv(run_dir / "primary_comparison.csv", index=False, encoding="utf-8-sig")
    infeasible = policies[~policies.index.isin(fit.index)]
    infeasible.to_csv(run_dir / "infeasible_policies.csv", index=False, encoding="utf-8-sig")
    mechanisms = load_csv(run_dir / "mechanisms" / "mechanisms_metrics.csv")
    if not mechanisms.empty:
        mechanism_columns = [key for key in ["training_records", "train_thin_count", "train_thin_difficulty_rate", "auc", "brier",
                                             "thin_frr", "thick_frr", "abs_frr_gap", "approval_rate", "risk", "utility5"] if key in mechanisms]
        mechanism_summary = mechanisms.groupby("condition", sort=False)[mechanism_columns].agg(["mean", "std", "count"])
        mechanism_summary.columns = ["_".join(column) for column in mechanism_summary.columns]
        mechanism_summary.reset_index().to_csv(run_dir / "mechanisms_comparison.csv", index=False, encoding="utf-8-sig")
    intervals = load_csv(run_dir / "paired_intervals.csv")
    if not intervals.empty:
        interval_primary = intervals[(intervals.comparison == "fairness_cost") &
                                     intervals.metric.isin(["utility", "approval_rate", "thin_frr", "thick_frr", "abs_frr_gap"])]
        interval_primary.to_csv(run_dir / "paired_intervals_primary.csv", index=False, encoding="utf-8-sig")
    summary = {"models_with_metrics": int(models.model.nunique()) if not models.empty else 0,
               "model_seed_records": len(models), "policy_records": len(policies), "feasible_policy_records": len(fit),
               "infeasible_or_unfinished_policy_records": len(infeasible), "matched_primary_comparisons": len(pairs),
               "completed_figures": len(manifest["figures"])}
    write_json(run_dir / "report_summary.json", summary)
    title = "V3 公平信贷与业务收益：当前运行结果\n"
    paragraphs = [title,
        f"目前已有 {summary['models_with_metrics']} 类模型、{len(models)} 条模型种子结果、{len(policies)} 条策略记录；其中 {len(fit)} 条具有可计算业务结果，{len(infeasible)} 条不可行或尚未完成。只解读已经保存的实际结果。完整完成情况以 status.json 和 verification.json 为准。",
        "本报告把 TARGET=1 解释为历史还款困难标签。困难标签并非实际本金损失；收益、损失率及资金成本属于明确设定的业务情景。单位净贡献等于获批正常标签数减去损失比乘以获批困难标签数。金额模型使用数据原始金额单位，不能直接当作人民币实际利润。",
        "主要比较采用同一模型、同一种子、同一风险预算和资金预算的 group_profit 与 group_profit_gap，避免把模型能力变化误当成公平机制的效果。标准统一阈值审批另列为业务参照。公平和风险约束在验证集上选择，测试集上的超出需要如实报告。",
        "模型精度图只反映统一数据与特征下本地完成的训练结果。公开论文的原始数据、划分或调参不同，因此这里不作达到 SOTA 的判断。AUC 与业务贡献分别评价，较高 AUC 不自动意味着较高利润。图中种子标准差描述实际训练差异；确定性模型在不同种子下可能完全相同，不能把重复结果当成额外独立证据。"]
    if not pairs.empty:
        for model, frame in pairs.groupby("model", sort=False):
            d_profit = frame.utility_fair - frame.utility_base
            d_approval = frame.approval_count_fair - frame.approval_count_base
            d_gap = (frame.abs_frr_gap_fair - frame.abs_frr_gap_base) * 100
            d_thin = (frame.thin_frr_fair - frame.thin_frr_base) * 100
            d_thick = (frame.thick_frr_fair - frame.thick_frr_base) * 100
            paragraphs.append(f"{model}：在已完成的 {len(frame)} 个种子中，加入主要公平约束后，平均批准数变化 {d_approval.mean():+.1f}，净贡献变化 {d_profit.mean():+.2f}，误拒差距变化 {d_gap.mean():+.2f} 个百分点；薄信用记录人群误拒率变化 {d_thin.mean():+.2f}，对照人群变化 {d_thick.mean():+.2f} 个百分点。需同时观察两组变化，不能只看差距变小。")
    else:
        paragraphs.append("主要风险预算、损失情景与公平容差下，尚无完整配对结果；因此目前不对本方法是否优于基线下结论。")
    if not prim.empty:
        constrained = prim[prim.family.ne("fairlearn")]
        exceed = (constrained.approved_risk > constrained.risk_budget + 1e-12).sum()
        fairness = constrained[constrained.epsilon.notna()]
        gap_exceed = (fairness.abs_frr_gap > fairness.epsilon + 1e-12).sum()
        paragraphs.append(f"主要情景的可行阈值策略中，测试集风险超过各自验证预算的记录有 {int(exceed)} 条；其中带公平容差的记录中，测试误拒差距超过容差的有 {int(gap_exceed)} 条。标准 Fairlearn 后处理单列比较，它本身没有加入这里的放贷风险和资金预算。这里报告泛化后的实际表现，验证约束并非测试保证。")
    if not infeasible.empty:
        paragraphs.append("不可行记录保存在 infeasible_policies.csv。不可行意味着当前模型、阈值搜索和预算组合没有找到满足全部条件的策略，应作为成本与可行性结果展示。未找到解的记录不进入已完成方案的均值。")
    if not amount.empty:
        paragraphs.append("金额业务模型见图 07：使用相同申请人群的请求本金作为资金预算分母，分别比较全部预算和收紧预算。金额净贡献使用相同收益、困难损失和成本情景，应在同一资金上限内比较公平审批与相应收益基线。金额大并不自动代表单位资金回报更高，完整记录也包含单位已放款本金的净贡献。")
    if not mechanisms.empty:
        labels = {"full": "完整训练数据", "random_same_total": "全体随机减少到相同总样本数",
                  "group_count_matched_preserve_label_mix": "各组样本数相同且保留原标签比例",
                  "biased_drop_thin_good": "选择性遗漏薄信用记录中的正常客户", "corrected_referenceweights": "代表性参考集重加权"}
        descriptions = []
        for condition, arm in mechanisms.groupby("condition", sort=False):
            descriptions.append(f"{labels.get(condition, condition)}：{len(arm)} 次采样，薄信用记录组误拒率均值 {arm.thin_frr.mean():.2%}，对照组 {arm.thick_frr.mean():.2%}，AUC {arm.auc.mean():.4f}，单位净贡献 {arm.utility5.mean():,.1f}")
        paragraphs.append("偏见机制对照已经完成的结果：" + "；".join(descriptions) + "。这些对照分别改变样本总量、组别数量和组内标签构成，可用于区分人数不足与标签选择偏差；重加权依赖代表性且有标签的参考集。完整均值与种子标准差见 mechanisms_comparison.csv。")
    if not intervals.empty:
        paragraphs.append("配对区间见 paired_intervals_primary.csv：它们对同一批测试申请人进行配对重采样，条件于已经训练好的模型和已经冻结的审批规则。区间衡量这批测试样本的重采样变化；训练种子差异另外报告，不能把二者合并解释为完整训练不确定性。")
    if not business_policies.empty:
        paragraphs.append("为提高业务比较的对齐程度，另加入成本敏感的标准 Fairlearn 基线：只在验证数据中重复困难标签记录，使加权准确率对应收益型目标，同时保持正常标签客户的组内 TPR 不变；再按原人群的验证收益、风险与批准下限选择成本权重。算法仍使用原始 ThresholdOptimizer。候选权重与验证结果全部保存在 fairlearn_business/candidate_validation.csv，选择后冻结规则，再计算测试结果。它的指标为随机化审批的期望，不能解释成一次实际抽样放贷的风险保证。")
    paragraphs += ["建议汇报顺序：用数据偏差对照解释小型机器学习的误拒来源；用收益分解说明公平约束改变了哪些放贷决定；再展示强模型与收益导向审批能否在相近公平水平下保留更多净贡献。",
                   "图表都由保存的结果重新生成，不在报告阶段训练或选阈值。figure_data/ 下的 CSV 保留各点的模型、种子、策略及源文件行号；figure_manifest.json 保存图片与源表的校验值。",
                   "历史测试集在此前项目中已经查看，本轮采用预先固定的选择规则进行扩展评价，不宣称全新盲测。"]
    report = "\n\n".join(paragraphs) + "\n"
    (run_dir / "RESULTS_CN.txt").write_text(report, encoding="utf-8")
    display_cols = [key for key in ["model", "seed", "family", "epsilon", "approval_rate", "approved_risk", "abs_frr_gap", "utility"] if key in selected]
    preview = selected[display_cols].copy()
    for col in preview.select_dtypes(include="number"):
        preview[col] = preview[col].round(5)
    (run_dir / "RESULTS_CN.md").write_text(report + "\n主要情景对照（完整结果见 CSV）\n\n" + table_md(preview) + "\n", encoding="utf-8")
    write_json(run_dir / "figure_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE / "results" / "run_001")
    args = parser.parse_args()
    result = main(args.run_dir)
    print(json.dumps({"figures": len(result["figures"]), "skipped": result["skipped"]}, ensure_ascii=False))
