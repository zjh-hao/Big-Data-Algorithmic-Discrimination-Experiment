"""Reproduce an actual training-loss intervention on fixed training records.

The opportunity-weighted BCE is reused verbatim from the project's published
audit_extension/run_experiment.py, not simulated by changing test thresholds.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import HERE, ROOT, dump, load_config, now, sha
from .prepare import load_prepared
from .mechanisms import execute
from src.experiment import biased_indices


def load_original_loss_weights():
    source = ROOT / "audit_extension" / "run_experiment.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "loss_weights"]
    if len(nodes) != 1:
        raise ValueError("Original loss_weights definition missing or ambiguous")
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["loss_weights"], source


def validate_loss_equivalence(weights_fn):
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0])
    g = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    loss = np.linspace(0.1, 0.8, len(y))
    for lam in (0.0, 0.5, 1.0, 4.0):
        w = weights_fn(y, g, 1, lam)
        raw_w = np.ones(len(y))
        declared = loss.mean()
        for group in (0, 1):
            cell = (g == group) & (y == 0)
            raw_w[cell] += lam * len(y) / (2 * cell.sum())
            declared += lam / 2 * loss[cell].mean()
        np.testing.assert_allclose(w, raw_w / raw_w.mean())
        np.testing.assert_allclose(np.mean(w * loss) * raw_w.mean(), declared)
        assert np.isclose(w.mean(), 1)


def objective_summary(output, standard_rows):
    rows = list(standard_rows)
    for path in output.glob("*/status.json"):
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("status") != "PASS":
            continue
        meta = status["sample_design"]
        rows.append(dict(**status["metrics"], dataset=meta["dataset"], loss="opportunity",
                         opportunity_lambda=meta["opportunity_lambda"],
                         loss_source_sha256=meta["loss_source_sha256"],
                         source_status=str(path)))
    frame = pd.DataFrame(rows).sort_values(["dataset", "opportunity_lambda", "sampling_seed"])
    frame.to_csv(output / "training_objective_metrics.csv", index=False)
    columns = ["thin_frr", "thick_frr", "abs_frr_gap", "signed_frr_gap", "auc",
               "approval_rate", "risk", "good_approved", "difficulty_count", "utility5"]
    summary = frame.groupby(["dataset", "opportunity_lambda"])[columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(output / "summary.csv")
    changes = []
    for dataset in frame.dataset.unique():
        standard = frame[(frame.dataset == dataset) & (frame.opportunity_lambda == 0)]
        for lam in sorted(frame.opportunity_lambda.unique()):
            weighted = frame[(frame.dataset == dataset) & (frame.opportunity_lambda == lam)]
            if len(weighted) == 0:
                continue
            paired = weighted.merge(standard, on="sampling_seed", suffixes=("_new", "_base"))
            changes.append(dict(dataset=dataset, opportunity_lambda=lam, paired_runs=len(paired),
                thin_frr_change_pp=float(((paired.thin_frr_new - paired.thin_frr_base) * 100).mean()),
                thick_frr_change_pp=float(((paired.thick_frr_new - paired.thick_frr_base) * 100).mean()),
                gap_change_pp=float(((paired.abs_frr_gap_new - paired.abs_frr_gap_base) * 100).mean()),
                utility5_change=float((paired.utility5_new - paired.utility5_base).mean()),
                auc_change=float((paired.auc_new - paired.auc_base).mean())))
    pd.DataFrame(changes).to_csv(output / "changes_vs_standard.csv", index=False)
    make_figure(summary, output)
    return frame


def make_figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, dataset in enumerate(("full", "biased")):
        sub = summary.loc[dataset]
        index = np.arange(len(sub))
        label = ["Standard" if float(lam) == 0 else f"Opportunity {lam:g}" for lam in sub.index]
        for key, offset, color, title in [("thin_frr", -0.17, "#e07a38", "Thin"),
                                          ("thick_frr", 0.17, "#3574a8", "Thick")]:
            axes[row, 0].bar(index + offset, sub[f"{key}_mean"] * 100, 0.34,
                yerr=sub[f"{key}_std"].fillna(0) * 100, color=color, label=title, capsize=3)
        axes[row, 0].set_ylabel(f"{dataset.title()} data: normal-label FRR (%)")
        axes[row, 0].legend(frameon=False)
        for col, key, title, scale in [(1, "abs_frr_gap", "Absolute FRR gap (percentage points)", 100),
                                       (2, "utility5", "Scenario utility (normal - 5 x difficulty approvals)", 1)]:
            axes[row, col].bar(index, sub[f"{key}_mean"] * scale,
                yerr=sub[f"{key}_std"].fillna(0) * scale, color="#75998c", capsize=3)
            axes[row, col].set_ylabel(title)
        for ax in axes[row]:
            ax.set_xticks(index, label, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.18)
            ax.set_axisbelow(True)
    fig.suptitle("Changing the training loss on identical records: opportunity weighting", fontsize=15)
    fig.savefig(output / "training_objective_mechanism.png", dpi=180)
    fig.savefig(output / "training_objective_mechanism.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(HERE / "results" / "run_001" / "training_objective"))
    parser.add_argument("--lambdas", default="0.5,1,4")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    lambdas = [float(value) for value in args.lambdas.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else cfg["sampling_seeds"]
    if any(value <= 0 for value in lambdas):
        raise ValueError("Positive opportunity weights required; standard models are reused")
    loss_weights, source = load_original_loss_weights()
    validate_loss_equivalence(loss_weights)
    identity = dict(code_sha256=sha(__file__), helper_code_sha256=sha(HERE / "mechanisms.py"),
                    loss_source_sha256=sha(source), lambdas=lambdas, sampling_seeds=seeds,
                    formula="L=(1/n) sum_i BCE_i + (lambda/2) sum_g mean_{i:g,TARGET=0} BCE_i",
                    equivalent_weight="w_i=1+lambda*n/(2*n_g0)*I(TARGET_i=0), then divide by mean(w)",
                    interpretation="Both normal-label groups receive an equal-group opportunity term; not a hard FRR-gap constraint",
                    approval_objective="Unchanged: validation maximum common-threshold approval count at risk <= 5%",
                    training_weight_validation="PASS: algebraic loss/weight equivalence and mean-one weights",
                    source_file=str(source))
    if (output / "design.json").exists():
        existing = json.loads((output / "design.json").read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("Objective experiment code/config changed; select a new output directory")
    dump(output / "design.json", identity)
    data = load_prepared(cfg)
    records, split = data[2], data[3]
    y, g = records.TARGET.to_numpy(), records.thin.to_numpy()
    tr = split["train"]
    baselines = [("full", 0, np.arange(len(tr)), "full_sampling0")]
    baselines.extend(("biased", seed, biased_indices(y[tr], g[tr], cfg["sampling_retention"], seed),
                      f"biased_drop_thin_good_sampling{seed}") for seed in seeds)
    standard_rows = []
    mechanism_folder = output.parent / "mechanisms"
    for dataset, seed, used, name in baselines:
        path = mechanism_folder / name / "status.json"
        source_status = json.loads(path.read_text(encoding="utf-8"))
        if source_status["status"] != "PASS":
            raise ValueError("Standard training baseline not complete")
        standard_rows.append(dict(**source_status["metrics"], dataset=dataset, loss="standard",
                                  opportunity_lambda=0.0, source_status=str(path),
                                  source_status_sha256=sha(path), loss_source_sha256=sha(source)))
    dump(output / "reused_standard_baselines.json", standard_rows)
    started = time.perf_counter()
    with threadpool_limits(limits=min(4, cfg["threads"])):
        for dataset, seed, used, _ in baselines:
            selected = tr[used]
            for lam in lambdas:
                weights = loss_weights(y[selected], g[selected], 1, lam)
                assert np.isclose(weights.mean(), 1)
                name = f"{dataset}_opportunity{lam:g}".replace(".", "p")
                notes = dict(dataset=dataset, opportunity_lambda=lam, class_difficulty_cost=1,
                             source_file=str(source), loss_source_sha256=sha(source),
                             training_loss_code_sha256=sha(__file__), formula=identity["formula"],
                             equivalent_weight=identity["equivalent_weight"])
                print("Starting training loss", name, "sampling seed", seed, flush=True)
                execute(name, seed, used, weights, notes, data, cfg, output, output.parent)
                objective_summary(output, standard_rows)
    frame = objective_summary(output, standard_rows)
    dump(output / "completion.json", dict(status="PASS", completed_utc=now(),
         new_training_jobs=len(baselines) * len(lambdas), reused_standard_jobs=len(standard_rows),
         total_comparison_rows=len(frame), wall_seconds=time.perf_counter() - started,
         **identity))


if __name__ == "__main__":
    main()
