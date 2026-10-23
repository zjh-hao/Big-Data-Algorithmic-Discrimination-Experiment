"""Validate paired resampling against explicit row-level calculations."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research_extension_v3.business import fit_policies
from research_extension_v3.inference import (
    _aligned, _cell_metrics, _cells, build_inference, paired_bootstrap, subgroup_metrics,
)


def data():
    rows = []
    for group in [0, 1]:
        for band in [0, 1, 2]:
            for j, y in enumerate([0, 0, 1, 0]):
                i = len(rows)
                rows.append({"SK_ID_CURR": i + 100, "TARGET": y, "thin": group,
                             "income_band": band, "probability": [.02, .12, .18, .4][j] + group * .02,
                             "AMT_CREDIT": 100 + 20 * i})
    frame = pd.DataFrame(rows)
    base = np.tile([1, 0, 1, 1], 6)
    cand = np.tile([1, 1, 0, 1], 6)
    return frame, base, cand


def direct_metrics(frame, a, loss=3):
    y, g = frame.TARGET.to_numpy(), frame.thin.to_numpy()
    good, bad = np.sum(a * (1 - y)), np.sum(a * y)
    result = {"utility": good - loss * bad,
              "utility_per_1000_applicants": (good - loss * bad) * 1000 / len(frame),
              "approvals": a.sum(), "approval_rate": a.mean(),
              "good_approved": good, "difficulty_count": bad,
              "risk": bad / a.sum() if a.sum() else np.nan}
    for h, name in [(0, "thick"), (1, "thin")]:
        normal = (g == h) & (y == 0)
        result[name + "_frr"] = np.mean(1 - a[normal]) if normal.any() else np.nan
    result["abs_frr_gap"] = abs(result["thin_frr"] - result["thick_frr"])
    return result


class InferenceTests(unittest.TestCase):
    def test_collapsed_cells_equal_explicit_paired_row_resamples(self):
        frame, base, cand = data()
        cells, _ = _cells(frame, base, cand)
        record_cells = np.column_stack([frame.thin, frame.TARGET, base, cand, frame.income_band])
        samples = np.random.default_rng(16).integers(0, len(frame), size=(80, len(frame)))
        counts = np.array([[np.sum(np.all(record_cells[index] == cell, axis=1)) for cell in cells]
                           for index in samples])
        collapsed_b = _cell_metrics(counts, cells, 2, 3)
        collapsed_c = _cell_metrics(counts, cells, 3, 3)
        for i, indices in enumerate(samples):
            expected_b = direct_metrics(frame.iloc[indices], base[indices])
            expected_c = direct_metrics(frame.iloc[indices], cand[indices])
            for name in expected_b:
                np.testing.assert_allclose(collapsed_b[name][i], expected_b[name], equal_nan=True)
                np.testing.assert_allclose(collapsed_c[name][i], expected_c[name], equal_nan=True)

    def test_identical_frozen_decisions_have_exactly_zero_intervals(self):
        frame, base, _ = data()
        rows, subgroups = paired_bootstrap(frame, base, base, repetitions=300, seed=1)
        for row in rows + subgroups:
            self.assertEqual(row["delta"], 0)
            self.assertEqual(row["ci_low"], 0)
            self.assertEqual(row["ci_high"], 0)
            self.assertEqual(row["interval_status"], "PASS")

    def test_point_deltas_match_independent_payoff_and_frr(self):
        frame, base, cand = data()
        rows, parts = paired_bootstrap(frame, base, cand, loss_ratio=3, repetitions=500, seed=5)
        by_metric = {r["metric"]: r for r in rows}
        self.assertEqual(by_metric["utility"]["baseline"], -6)
        self.assertEqual(by_metric["utility"]["candidate"], 18)
        self.assertEqual(by_metric["utility"]["delta"], 24)
        self.assertEqual(by_metric["approvals"]["delta"], 0)
        self.assertAlmostEqual(by_metric["thin_frr"]["delta"], -1 / 3)
        self.assertAlmostEqual(by_metric["thick_frr"]["delta"], -1 / 3)
        self.assertEqual(by_metric["abs_frr_gap"]["delta"], 0)
        self.assertEqual(len(parts), 6 * 8)
        self.assertTrue(by_metric["utility"]["ci_low"] > 0)

    def test_seed_is_reproducible(self):
        frame, base, cand = data()
        first = paired_bootstrap(frame, base, cand, repetitions=101, seed=19)
        second = paired_bootstrap(frame, base, cand, repetitions=101, seed=19)
        for a, b in zip(first, second):
            pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))

    def test_subgroup_points_include_amount_and_correct_denominators(self):
        frame, base, _ = data()
        groups = subgroup_metrics(frame, base, loss_ratio=3)
        self.assertEqual(len(groups), 6)
        first = groups[0]
        self.assertEqual(first["records"], 4)
        self.assertEqual(first["normal_records"], 3)
        self.assertAlmostEqual(first["frr"], 1 / 3)
        self.assertEqual(first["utility"], -1)
        self.assertEqual(first["approved_amount"], 100 + 140 + 160)
        self.assertAlmostEqual(first["amount_utility"], .09 * (100 + 160 - 3 * 140))

    def test_missing_subgroup_has_explicit_no_support(self):
        frame, base, cand = data()
        mask = frame.income_band.to_numpy() == 0
        _, parts = paired_bootstrap(frame.loc[mask], base[mask], cand[mask], repetitions=100)
        absent = [r for r in parts if r["income_band"] != 0]
        self.assertTrue(all(r["interval_status"] == "NO_SUPPORT" for r in absent))
        self.assertTrue(all(np.isnan(r["ci_low"]) for r in absent))

    def test_pairing_reorders_ids_and_rejects_disagreeing_records(self):
        frame, _, _ = data()
        a, b = _aligned(frame, frame.sample(frac=1, random_state=11))
        pd.testing.assert_frame_equal(a, b)
        changed = frame.copy()
        changed.loc[0, "TARGET"] = 1
        with self.assertRaises(ValueError):
            _aligned(frame, changed)

    def test_fractional_decisions_are_rejected_for_cell_bootstrap(self):
        frame, base, cand = data()
        cand = cand.astype(float)
        cand[0] = .5
        with self.assertRaises(ValueError):
            paired_bootstrap(frame, base, cand)
        with self.assertRaises(ValueError):
            paired_bootstrap(frame, base, base, repetitions=1)

    def test_saved_run_outputs_and_no_test_selection(self):
        frame, _, _ = data()
        cfg = {"grid": 16, "minimum_approvals": 1, "primary_loss_ratio": 3,
               "primary_risk_budget": .4, "primary_epsilon": .02, "epsilons": [.02, .01],
               "loss_ratios": [3], "risk_budgets": [.4], "capital_fractions": [1],
               "noharm_deltas": [.01], "bootstrap_repetitions": 100}
        frozen = fit_policies(frame, cfg)
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            for part in ["jobs", "predictions", "policies"]:
                (run / part).mkdir()
            (run / "config_frozen.json").write_text(json.dumps(cfg), encoding="utf-8")
            for model, seed in [("xgboost", 731), ("lightgbm", 732)]:
                name = f"{model}_s{seed}"
                (run / "jobs" / f"{name}.json").write_text(json.dumps({"model": model, "seed": seed, "status": "PASS"}), encoding="utf-8")
                (run / "policies" / f"{name}.json").write_text(json.dumps(frozen), encoding="utf-8")
                frame.to_parquet(run / "predictions" / f"{name}_test.parquet", index=False)
            manifest = build_inference(run, seed=11)
            self.assertEqual(len(manifest["completed_jobs"]), 2)
            for name in ["paired_intervals.csv", "subgroups.csv", "paired_subgroup_intervals.csv", "inference_manifest.json"]:
                self.assertTrue((run / name).is_file())
            intervals = pd.read_csv(run / "paired_intervals.csv")
            self.assertIn("fairness_cost_epsilon_1pct", intervals.comparison.to_list())
            cross = intervals[intervals.comparison == "model_gain_after_fairness_vs_xgboost_s731"]
            self.assertEqual(set(cross.model), {"xgboost", "lightgbm"})


if __name__ == "__main__":
    unittest.main()
