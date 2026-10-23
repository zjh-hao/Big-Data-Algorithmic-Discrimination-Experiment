"""Small independent oracles for scenario payoffs and constrained selection."""
import itertools
import json
import unittest

import numpy as np
import pandas as pd

from research_extension_v3.business import (
    apply_policy, evaluate_policies, fit_policies, metrics_for_decisions,
)


def fixture():
    # Both groups include high-risk normal borrowers and low-risk difficulties.
    return pd.DataFrame({
        "SK_ID_CURR": range(10, 22),
        "TARGET": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "thin": [0] * 6 + [1] * 6,
        "probability": [.02, .07, .11, .18, .38, .80, .01, .08, .12, .31, .62, .90],
        "AMT_CREDIT": [100, 300, 200, 250, 80, 900, 130, 400, 600, 170, 90, 110],
    })


def config():
    return {"grid": 32, "minimum_approvals": 2,
            "primary_risk_budget": .30, "risk_budgets": [.30],
            "primary_loss_ratio": 3, "loss_ratios": [3],
            "primary_epsilon": .25, "epsilons": [.5, .25, 0.0],
            "capital_fractions": [1.0, .6], "noharm_deltas": [0.0, .25],
            "scenario_interest": .12, "scenario_cost": .03}


def oracle(frame, grid, policy, minimum, baseline=None):
    """Exhaustive scalar search independent of the cumulative engine."""
    y = frame.TARGET.to_numpy()
    p = frame.probability.to_numpy()
    g = frame.thin.to_numpy()
    amount = frame.AMT_CREDIT.to_numpy()
    normal = [int(np.sum((y == 0) & (g == h))) for h in [0, 1]]
    candidates = []
    for i, j in itertools.product(range(len(grid)), repeat=2):
        if policy["family"] == "common_profit" and i != j:
            continue
        a = p <= np.where(g == 0, grid[i], grid[j])
        n = int(a.sum())
        if n < minimum:
            continue
        bad = int(np.sum(a & (y == 1)))
        approved_amount = float(amount[a].sum())
        if bad > policy["risk_budget"] * n + 1e-10:
            continue
        if approved_amount > policy["capital_fraction"] * amount.sum() + 1e-8:
            continue
        frr = [1 - int(np.sum(a & (g == h) & (y == 0))) / normal[h] for h in [0, 1]]
        gap = abs(frr[0] - frr[1])
        if policy["epsilon"] is not None and gap > policy["epsilon"] + 1e-10:
            continue
        if policy["noharm_delta"] is not None:
            if baseline is None or not baseline["feasible"]:
                continue
            bm = baseline["validation_metrics"]
            if any(frr[h] > bm[name + "_frr"] + policy["noharm_delta"] + 1e-10
                   for h, name in [(0, "thick"), (1, "thin")]):
                continue
        good = n - bad
        unit = good - policy["loss_ratio"] * bad
        # At .12 income/.03 cost, difficulty total net loss is lambda * .09.
        weighted = .09 * float(np.sum(amount[a] * np.where(y[a] == 0, 1, -policy["loss_ratio"])))
        value = unit if policy["utility_basis"] == "unit_utility" else weighted
        if policy["objective"] == "approvals":
            key = (-n, -value, bad / n, gap, i, j)
        else:
            key = (-value, -n, bad / n, gap, i, j)
        candidates.append((key, [float(grid[i]), float(grid[j])], value, n))
    return min(candidates, key=lambda x: x[0]) if candidates else None


class BusinessPolicyTests(unittest.TestCase):
    def test_payoff_and_frr_are_computed_for_normal_labels(self):
        frame = fixture()
        a = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        m = metrics_for_decisions(frame, a, loss_ratio=3)
        self.assertEqual(m["approvals"], 5)
        self.assertEqual(m["difficulty_count"], 1)
        self.assertEqual(m["good_approved"], 4)
        self.assertEqual(m["utility"], 1)
        self.assertAlmostEqual(m["thick_frr"], .25)
        self.assertAlmostEqual(m["thin_frr"], .75)
        self.assertAlmostEqual(m["abs_frr_gap"], .5)
        self.assertAlmostEqual(m["amount_utility"], .09 * (100 + 300 + 250 + 130 - 3 * 400))
        self.assertAlmostEqual(m["amount_weighted_risk"], 400 / (100 + 300 + 250 + 130 + 400))

    def test_fractional_randomized_decisions_use_expected_counts(self):
        frame = fixture().iloc[[0, 2, 6, 7]].copy()
        m = metrics_for_decisions(frame, [.5, .25, 1, 0], loss_ratio=5)
        self.assertEqual(m["decision_type"], "stochastic_expectation")
        self.assertAlmostEqual(m["approvals"], 1.75)
        self.assertAlmostEqual(m["risk"], .25 / 1.75)
        self.assertAlmostEqual(m["utility"], 1.5 - 5 * .25)
        self.assertAlmostEqual(m["thick_frr"], .5)
        self.assertAlmostEqual(m["thin_frr"], 0)

    def test_all_nonprotected_policies_match_exhaustive_optimum(self):
        frame, cfg = fixture(), config()
        frozen = fit_policies(frame, cfg)
        grid = frozen["metadata"]["threshold_grid"]
        # On this tiny sample the grid includes every distinct decision set.
        self.assertTrue(set(frame.probability).issubset(grid))
        checked = 0
        for policy in frozen["policies"]:
            if policy["noharm_delta"] is not None:
                continue
            expected = oracle(frame, grid, policy, cfg["minimum_approvals"])
            self.assertEqual(policy["feasible"], expected is not None, policy["policy_id"])
            if expected is not None:
                self.assertEqual(policy["thresholds"], expected[1], policy["policy_id"])
                expected_value = expected[3] if policy["objective"] == "approvals" else expected[2]
                self.assertAlmostEqual(policy["validation_objective"], expected_value)
            checked += 1
        self.assertGreater(checked, 10)

    def test_constraints_hold_on_validation_and_nested_profit(self):
        cfg, frame = config(), fixture()
        frozen = fit_policies(frame, cfg)
        by_id = {p["policy_id"]: p for p in frozen["policies"]}
        for p in frozen["policies"]:
            if not p["feasible"]:
                self.assertIsNone(p["thresholds"])
                continue
            m = p["validation_metrics"]
            self.assertLessEqual(m["risk"], p["risk_budget"] + 1e-10)
            self.assertLessEqual(m["capital_fraction_used"], p["capital_fraction"] + 1e-10)
            self.assertGreaterEqual(m["approvals"], cfg["minimum_approvals"])
            if p["epsilon"] is not None:
                self.assertLessEqual(m["abs_frr_gap"], p["epsilon"] + 1e-10)
            if p["noharm_delta"] is not None:
                baseline = by_id[p["baseline_policy_id"]]
                for group in ["thin", "thick"]:
                    self.assertLessEqual(m[group + "_frr"], baseline["validation_metrics"][group + "_frr"] + p["noharm_delta"] + 1e-10)
        for basis in ["unit_utility", "amount_utility"]:
            subset = [p for p in frozen["policies"] if p["utility_basis"] == basis and p["capital_fraction"] == 1]
            baseline = next(p for p in subset if p["family"] == "group_profit")
            if baseline["feasible"]:
                for p in subset:
                    if p["family"].startswith("group_profit_gap") and p["feasible"]:
                        self.assertLessEqual(p["validation_objective"], baseline["validation_objective"] + 1e-8)

    def test_protected_policy_is_exhaustive_optimum_or_infeasible(self):
        frame, cfg = fixture(), config()
        frozen = fit_policies(frame, cfg)
        by_id = {p["policy_id"]: p for p in frozen["policies"]}
        for policy in frozen["policies"]:
            if policy["noharm_delta"] is None:
                continue
            baseline = by_id[policy["baseline_policy_id"]]
            expected = oracle(frame, frozen["metadata"]["threshold_grid"], policy,
                              cfg["minimum_approvals"], baseline=baseline)
            self.assertEqual(policy["feasible"], expected is not None)
            if expected is not None:
                self.assertEqual(policy["thresholds"], expected[1])

    def test_expected_payoffs_depend_on_probabilities_not_historical_labels(self):
        frame = fixture()
        decisions = np.ones(len(frame))
        first = metrics_for_decisions(frame, decisions, loss_ratio=5)
        frame["TARGET"] = 1 - frame.TARGET
        second = metrics_for_decisions(frame, decisions, loss_ratio=5)
        self.assertEqual(first["expected_utility"], second["expected_utility"])
        self.assertEqual(first["expected_amount_utility"], second["expected_amount_utility"])
        self.assertNotEqual(first["utility"], second["utility"])

    def test_test_labels_never_change_approval_decisions_or_frozen_ledger(self):
        frame, cfg = fixture(), config()
        frozen = fit_policies(frame, cfg)
        before = json.dumps(frozen, sort_keys=True, allow_nan=False)
        other = frame.copy()
        other["TARGET"] = 1 - other.TARGET
        first, second = evaluate_policies(frame, frozen, cfg), evaluate_policies(other, frozen, cfg)
        for a, b in zip(first, second):
            self.assertEqual(a["policy_id"], b["policy_id"])
            self.assertEqual(a["thresholds"], b["thresholds"])
            if a["feasible"]:
                self.assertEqual(a["approvals"], b["approvals"])
                self.assertEqual(a["approved_amount"], b["approved_amount"])
        self.assertEqual(before, json.dumps(frozen, sort_keys=True, allow_nan=False))
        self.assertTrue(any(a.get("risk") != b.get("risk") for a, b in zip(first, second) if a["feasible"]))

    def test_violated_test_constraints_are_reported_without_reoptimization(self):
        frame, cfg = fixture(), config()
        frozen = fit_policies(frame, cfg)
        changed = frame.copy()
        # Preserve normal support while making most accepted cases difficulties.
        changed["TARGET"] = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
        rows = evaluate_policies(changed, frozen, cfg)
        self.assertTrue(any(r.get("test_risk_constraint_met") is False for r in rows if r["feasible"]))
        self.assertEqual([r["thresholds"] for r in rows], [p["thresholds"] for p in frozen["policies"]])

    def test_infeasible_policies_remain_explicit(self):
        cfg = config()
        cfg["minimum_approvals"] = 999
        frozen = fit_policies(fixture(), cfg)
        self.assertEqual(frozen["metadata"]["feasible_count"], 0)
        self.assertTrue(all(p["status"] == "INFEASIBLE" and p["thresholds"] is None for p in frozen["policies"]))
        rows = evaluate_policies(fixture(), frozen, cfg)
        self.assertEqual(len(rows), len(frozen["policies"]))
        self.assertTrue(all("utility" not in r and r["reason"] for r in rows))

    def test_json_roundtrip_and_row_order_invariance(self):
        frame, cfg = fixture(), config()
        frozen = fit_policies(frame, cfg)
        restored = json.loads(json.dumps(frozen, allow_nan=False))
        evaluate_policies(frame, restored, cfg)
        shuffled = fit_policies(frame.sample(frac=1, random_state=91), cfg)
        self.assertEqual(frozen["metadata"]["validation_id_sha256"], shuffled["metadata"]["validation_id_sha256"])
        self.assertEqual([p["thresholds"] for p in frozen["policies"]], [p["thresholds"] for p in shuffled["policies"]])

    def test_reject_all_policy_has_defined_empty_payoffs(self):
        frame = fixture()
        policy = {"feasible": True, "thresholds": [-1, -1]}
        decisions = apply_policy(frame, policy)
        self.assertFalse(decisions.any())
        m = metrics_for_decisions(frame, decisions)
        self.assertEqual(m["utility"], 0)
        self.assertEqual(m["amount_utility"], 0)
        self.assertIsNone(m["risk"])
        self.assertEqual(m["thin_frr"], 1)

    def test_invalid_inputs_fail_loudly(self):
        for column, value in [("probability", 1.1), ("AMT_CREDIT", -1), ("TARGET", 2), ("thin", 4)]:
            frame = fixture()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                fit_policies(frame, config())
        frame = fixture()
        frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]
        with self.assertRaises(ValueError):
            fit_policies(frame, config())
        with self.assertRaises(ValueError):
            metrics_for_decisions(fixture(), np.ones(2))
        with self.assertRaises(ValueError):
            apply_policy(fixture(), {"feasible": False, "thresholds": None})

    def test_sampling_sensitivity_is_one_parameter_at_a_time(self):
        cfg = config()
        cfg.update({"risk_budgets": [.1, .3, .5], "loss_ratios": [2, 3, 5]})
        policies = fit_policies(fixture(), cfg)["policies"]
        unit_settings = {(p["loss_ratio"], p["risk_budget"]) for p in policies if p["utility_basis"] == "unit_utility"}
        self.assertEqual(unit_settings, {(3, .1), (3, .3), (3, .5), (2, .3), (5, .3)})


if __name__ == "__main__":
    unittest.main()
