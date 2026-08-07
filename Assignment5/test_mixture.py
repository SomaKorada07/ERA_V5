"""
Invariant tests for the V5 mixture spec.

Asserts every claim README.md makes about its own numbers. Run with either:

    python3 test_mixture.py          # plain stdlib, no pytest needed
    python3 -m pytest test_mixture.py

If any README number is edited into an inconsistent state, a test fails.
"""

from __future__ import annotations

import unittest

import mixture_spec as ms
from proxy_experiments import experiment_1, experiment_2, experiment_3


class TestBudget(unittest.TestCase):
    def test_main_plus_anneal_equals_total(self):
        self.assertAlmostEqual(ms.MAIN_B + ms.ANNEAL_B, ms.TOTAL_B)

    def test_anneal_is_five_percent(self):
        self.assertAlmostEqual(ms.ANNEAL_B / ms.TOTAL_B, 0.05)


class TestMainMixture(unittest.TestCase):
    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(l.share for l in ms.MAIN_LANES), 1.0)

    def test_tokens_sum_to_main_budget(self):
        self.assertAlmostEqual(sum(l.tokens() for l in ms.MAIN_LANES), ms.MAIN_B, places=3)

    def test_protected_floors(self):
        floors = {l.name: (l.share, l.floor) for l in ms.MAIN_LANES if l.floor}
        for name, (share, floor) in floors.items():
            self.assertGreaterEqual(share, floor, f"{name} below floor")

    def test_readme_token_counts(self):
        # Values quoted in README §2.
        expected = {
            "General web": 608, "Code": 475, "Indic": 323, "STEM/math": 247,
            "Reasoning traces": 114, "Long-context": 95, "Agentic/tools": 38,
        }
        for l in ms.MAIN_LANES:
            self.assertAlmostEqual(l.tokens(), expected[l.name], places=0)

    def test_agentic_is_almost_all_synthetic(self):
        agentic = next(l for l in ms.MAIN_LANES if l.name == "Agentic/tools")
        synthetic_frac = 1 - agentic.supply_b / agentic.tokens()
        self.assertGreater(synthetic_frac, 0.97)  # README claims ~98%

    def test_stem_has_no_repetition_margin(self):
        stem = next(l for l in ms.MAIN_LANES if l.name == "STEM/math")
        self.assertLessEqual(stem.repetition_factor(), 1.0)
        self.assertGreater(stem.repetition_factor(), 0.95)


class TestIndicTiers(unittest.TestCase):
    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(t.share for t in ms.INDIC_TIERS), 1.0)

    def test_tier_tokens_match_budget(self):
        total = ms.indic_total_b()
        self.assertAlmostEqual(sum(t.share * total for t in ms.INDIC_TIERS), total, places=3)

    def test_tier_a_reserve_covers_anneal_slot(self):
        # The flaw from the earlier draft: reserve must fit the slot that uses it.
        tier_a = next(t for t in ms.INDIC_TIERS if t.name.startswith("A"))
        anneal_indic = next(a for a in ms.ANNEAL_LANES if a.name == "Indic")
        self.assertGreaterEqual(tier_a.supply_b, anneal_indic.tokens())


class TestReasoningBands(unittest.TestCase):
    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(b.share for b in ms.REASONING_BANDS), 1.0)


class TestAnneal(unittest.TestCase):
    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(a.share for a in ms.ANNEAL_LANES), 1.0)

    def test_tokens_sum_to_anneal_budget(self):
        self.assertAlmostEqual(sum(a.tokens() for a in ms.ANNEAL_LANES), ms.ANNEAL_B, places=3)

    def test_reserve_only_slots_within_oversample_cap(self):
        # Reserve-only slots (Tier-A Indic, ultra reasoning) must fit unaided.
        for a in ms.ANNEAL_LANES:
            if a.reserve_supply_b and a.reserve_only:
                oversample = a.tokens() / a.reserve_supply_b
                self.assertLessEqual(
                    oversample, ms.MAX_OVERSAMPLE,
                    f"{a.name} needs {oversample:.1f}x oversampling",
                )

    def test_agentic_slot_is_reserve_plus_synthetic(self):
        # Agentic cannot be reserve-only: 0.63B real can't fill 15B.
        agentic = next(a for a in ms.ANNEAL_LANES if a.name == "Agentic")
        self.assertFalse(agentic.reserve_only)
        topup = agentic.synthetic_topup_b()
        self.assertGreater(topup, 0.0)          # synthetic top-up is required
        self.assertAlmostEqual(topup, 15.0 - 0.63 * 5.0, places=2)


class TestAllChecksPass(unittest.TestCase):
    def test_run_checks_all_pass(self):
        failed = [c for c in ms.run_checks() if not c.ok]
        self.assertEqual(failed, [], f"failing invariants: {[c.name for c in failed]}")


class TestProxyDecisions(unittest.TestCase):
    def test_exp1_floor_sufficient(self):
        self.assertEqual(experiment_1(44, 45, 45.5, 45, 44.5).verdict, "FLOOR_SUFFICIENT")

    def test_exp1_hold(self):
        self.assertEqual(experiment_1(41, 45, 46.5, 45, 43).verdict, "HOLD_17")

    def test_exp1_raise(self):
        self.assertEqual(experiment_1(41, 45, 49, 45, 43).verdict, "RAISE_INDIC")

    def test_exp1_trim_synthetic(self):
        self.assertEqual(experiment_1(41, 45, 46, 45, 44.6).verdict, "TRIM_SYNTHETIC")

    def test_exp2_reserve_confirmed(self):
        self.assertEqual(experiment_2(38, 46).verdict, "RESERVE_CONFIRMED")

    def test_exp2_reserve_not_worth(self):
        self.assertEqual(experiment_2(44, 45.5).verdict, "RESERVE_NOT_WORTH_IT")

    def test_exp3_keep_on_quality(self):
        self.assertEqual(experiment_3(182, 179, 3, 4).verdict, "KEEP_CURRICULUM")

    def test_exp3_keep_on_stability(self):
        self.assertEqual(experiment_3(180.5, 180, 1.5, 6).verdict, "KEEP_CURRICULUM")

    def test_exp3_simplify(self):
        self.assertEqual(experiment_3(180.2, 180, 3.9, 4).verdict, "SIMPLIFY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
