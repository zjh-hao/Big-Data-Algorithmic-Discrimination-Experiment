"""Independent CDF reconstruction and assumptions-aware certificate checks."""
from copy import deepcopy
import math
import unittest

import numpy as np
import pandas as pd

from research_extension_v3.theory_extension.paired_certificate import (
    aligned_weights, certificate_rows, policy_approval_probabilities,
)


def data():
    return pd.DataFrame([
        {"thin": g, "income_band": b, "TARGET": y, "probability": score}
        for g in (0, 1) for b in (0, 1, 2)
        for y, score in ((0, .1), (0, .5), (0, .9), (1, .3))
    ])


def policy(name, thresholds=(.2, .8), weights=(.6, .4)):
    return {"policy_id": name, "feasible": True, "epsilon": .02, "delta": .01,
            "mixture": {str(g): {"thresholds": list(thresholds), "weights": list(weights)}
                        for g in (0, 1)}}


def frozen(candidate=None):
    baseline = policy("lp_profit_baseline")
    baseline["is_lp_profit_baseline"] = True
    candidate = candidate if candidate is not None else policy("same")
    return {"policies": [baseline, candidate]}


class PairedCertificateTests(unittest.TestCase):
    def test_identical_weights_zero_relative_uncertainty(self):
        rows = certificate_rows(data(), frozen())
        cells = [r for r in rows if r["row_type"] == "normal_cell"]
        self.assertEqual(len(cells), 16)
        for row in cells:
            self.assertEqual(row["l1_weight_distance"], 0)
            self.assertEqual(row["empirical_harm"], 0)
            self.assertEqual(row["paired_halfwidth"], 0)
            self.assertEqual(row["harm_upper"], 0)
            self.assertGreater(row["dkw_radius"], 0)

    def test_merge_dictionary_does_not_confuse_positions_or_duplicates(self):
        baseline = policy("base", (.2, .8), (.6, .4))
        candidate = policy("different", (.1, .2, .2, .8), (0., .2, .4, .4))
        grid, w, v = aligned_weights(candidate, baseline, 0)
        np.testing.assert_array_equal(grid, [.1, .2, .8])
        np.testing.assert_allclose(w, v, atol=1e-15)
        np.testing.assert_allclose(policy_approval_probabilities(data(), candidate),
                                   policy_approval_probabilities(data(), baseline))

    def test_cdf_dot_product_matches_direct_pair_mean(self):
        frame = data()
        candidate = policy("changed", (.4, .95), (.7, .3))
        base = frozen(candidate)["policies"][0]
        rows = certificate_rows(frame, frozen(candidate), alpha=.05, number_of_scorers=9)
        row = next(r for r in rows if r["policy_id"] == "changed" and r["cell"] == "subgroup_1_2")
        x = frame.loc[(frame.TARGET == 0) & (frame.thin == 1) & (frame.income_band == 2), "probability"].to_numpy()
        grid, w, v = aligned_weights(candidate, base, 1)
        cdf = np.array([(x <= t).mean() for t in grid])
        self.assertAlmostEqual(row["empirical_harm"], np.dot(v-w, cdf))
        radius = math.sqrt(math.log(2 * 8 * 9 / .05) / (2*len(x)))
        self.assertAlmostEqual(row["dkw_radius"], radius)
        self.assertAlmostEqual(row["paired_halfwidth"], radius*np.abs(w-v).sum())

    def test_small_data_does_not_claim_tight_parity_certificate(self):
        rows = certificate_rows(data(), frozen())
        row = next(r for r in rows if r["policy_id"] == "same" and r["row_type"] == "macro_parity")
        self.assertEqual(row["empirical_abs_gap"], 0)
        self.assertGreater(row["gap_halfwidth"], .02)
        self.assertFalse(row["parity_certified"])
        self.assertEqual(row["simultaneous_cdf_events"], 72)
        self.assertFalse(row["certificate_is_optimization_constraint"])

    def test_more_scorers_widens_same_data_bound(self):
        one = certificate_rows(data(), frozen(), number_of_scorers=1)[0]
        nine = certificate_rows(data(), frozen(), number_of_scorers=9)[0]
        self.assertGreater(nine["dkw_radius"], one["dkw_radius"])

    def test_robust_baseline_flag_does_not_override_explicit_primary(self):
        fr = frozen()
        robust = policy("lp_profit_risk_robust", (.4, .8), (.6, .4))
        robust.update(is_lp_profit_baseline=True, baseline_policy_id="lp_profit_baseline")
        fr["policies"].append(robust)
        rows = certificate_rows(data(), fr)
        self.assertEqual(len(rows), 27)
        self.assertTrue(all(r["baseline_policy_id"] == "lp_profit_baseline" for r in rows))

    def test_frozen_application_never_uses_labels_or_income(self):
        p, frame = policy("frozen"), data()
        original = policy_approval_probabilities(frame, p)
        changed = frame.copy()
        changed["TARGET"] = 1-changed.TARGET
        changed["income_band"] = 99
        changed["AMT_CREDIT"] = -1
        np.testing.assert_array_equal(original, policy_approval_probabilities(changed, p))
        np.testing.assert_array_equal(original, policy_approval_probabilities(frame[["probability", "thin"]], p))

    def test_reordering_and_application_do_not_mutate_frozen_policy(self):
        fr = frozen(policy("new", (.9, .3), (.4, .6)))
        before = deepcopy(fr)
        frame = data()
        first = certificate_rows(frame, fr)
        second = certificate_rows(frame.sample(frac=1, random_state=9), fr)
        self.assertEqual(fr, before)
        for a, b in zip(first, second):
            self.assertEqual(a["cell"], b["cell"])
            for key in ("dkw_radius", "empirical_harm", "empirical_abs_gap"):
                if key in a and a[key] is not None:
                    self.assertAlmostEqual(a[key], b[key])

    def test_labels_select_normal_denominator_only(self):
        frame = data()
        candidate = policy("new", (.2, .95), (.4, .6))
        r1 = certificate_rows(frame, frozen(candidate))[0]
        frame.loc[(frame.thin == 0) & (frame.probability == .9), "TARGET"] = 1
        r2 = certificate_rows(frame, frozen(candidate))[0]
        self.assertEqual(r1["normal_records"], 9)
        self.assertEqual(r2["normal_records"], 6)
        self.assertGreater(r2["dkw_radius"], r1["dkw_radius"])

    def test_empty_cell_does_not_return_a_certificate(self):
        frame = data()
        frame.loc[(frame.thin == 1) & (frame.income_band == 2), "TARGET"] = 1
        row = next(r for r in certificate_rows(frame, frozen()) if r["cell"] == "subgroup_1_2")
        self.assertEqual(row["normal_records"], 0)
        self.assertIsNone(row["harm_upper"])
        self.assertIsNone(row["protection_certified"])

    def test_bad_scope_or_weights_fail_explicitly(self):
        for kwargs in ({"alpha": 1}, {"number_of_scorers": 0}, {"number_of_scorers": 1.2}):
            with self.assertRaises(ValueError):
                certificate_rows(data(), frozen(), **kwargs)
        frame = data().assign(split="test")
        with self.assertRaisesRegex(ValueError, "validation"):
            certificate_rows(frame, frozen())
        bad = frozen()
        bad["policies"][1]["mixture"]["0"]["weights"] = [-.1, 1.1]
        with self.assertRaisesRegex(ValueError, "convex"):
            certificate_rows(data(), bad)


if __name__ == "__main__":
    unittest.main()
