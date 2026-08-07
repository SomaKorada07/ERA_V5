"""
V5 data mixture — executable single source of truth.

Every number in README.md is defined here once and checked for internal
consistency. Running this file reproduces the README tables and prints a
supply/verdict report, so the spec cannot silently drift from its own claims.

    python3 mixture_spec.py            # full report
    python3 mixture_spec.py --check    # exit 1 if any invariant fails

All token quantities are in **billions (B) of tokens**.
No third-party dependencies — standard library only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Budget (§0)
# --------------------------------------------------------------------------
TOTAL_B = 2000.0            # 2T total budget
ANNEAL_B = 100.0            # final 5% carved out as a separate preset
MAIN_B = TOTAL_B - ANNEAL_B  # 1900B main pretraining run

# A lane's implied oversampling above this factor is treated as memorization,
# not learning — flagged as a failed check (see §6 agentic anneal note).
MAX_OVERSAMPLE = 8.0


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Lane:
    name: str
    share: float          # fraction of the main run (0..1)
    supply_b: float       # real unique tokens available (B)
    floor: float = 0.0    # protected always-on floor (fraction), 0 = none
    wins: str = ""        # benchmark(s) this lane targets

    def tokens(self, base_b: float = MAIN_B) -> float:
        return self.share * base_b

    def repetition_factor(self, base_b: float = MAIN_B) -> float:
        """How many times unique supply must be reused to meet demand.
        >1 means repetition needed; <1 means headroom."""
        if self.supply_b <= 0:
            return float("inf")
        return self.tokens(base_b) / self.supply_b


@dataclass(frozen=True)
class Tier:
    name: str
    share: float          # fraction of the total Indic budget
    supply_b: float
    note: str = ""


@dataclass(frozen=True)
class Band:
    name: str
    share: float          # fraction of the reasoning budget
    example: str = ""


@dataclass(frozen=True)
class AnnealLane:
    name: str
    share: float          # fraction of the anneal budget
    # Volume of the *reserved* source that feeds this slot, if the slot is
    # sourced from a scarce reserve (else None -> no reserve constraint).
    reserve_supply_b: float | None = None
    # reserve_only=True : the reserve alone must fill the slot (within
    #   MAX_OVERSAMPLE) -- e.g. Tier-A Indic, ultra reasoning.
    # reserve_only=False: the reserve is oversampled up to `real_oversample`
    #   and any shortfall is filled with top-quality synthetic -- e.g. agentic.
    reserve_only: bool = True
    real_oversample: float = 5.0
    concentrates: str = ""

    def tokens(self) -> float:
        return self.share * ANNEAL_B

    def synthetic_topup_b(self) -> float:
        """For non-reserve-only slots: synthetic tokens needed after the real
        reserve is oversampled at `real_oversample`. 0 for reserve-only slots."""
        if self.reserve_supply_b is None or self.reserve_only:
            return 0.0
        real_contribution = self.reserve_supply_b * self.real_oversample
        return max(0.0, self.tokens() - real_contribution)


# --------------------------------------------------------------------------
# The spec (mirrors README §2, §3, §5, §6)
# --------------------------------------------------------------------------

# §2 — main pretraining mixture (shares of MAIN_B)
MAIN_LANES: list[Lane] = [
    Lane("General web",      0.32, 4500.0,             wins="MMLU, HLE"),
    Lane("Code",             0.25, 1100.0,             wins="SWE-bench, LiveCodeBench"),
    Lane("Indic",            0.17,  250.0, floor=0.12, wins="MILU, IndicGenBench"),
    Lane("STEM/math",        0.13,  250.0,             wins="AIME, GPQA, MATH"),
    Lane("Reasoning traces", 0.06,   85.0, floor=0.05, wins="AIME, GPQA (via RLVR)"),
    Lane("Long-context",     0.05,  100.0,             wins="RULER, long-eval"),
    Lane("Agentic/tools",    0.02,    0.63, floor=0.02, wins="BFCL, tau-bench, GAIA"),
]

# §3 — Indic tier split (shares of the *total* Indic budget across the run).
# Tier A is reserved for the anneal, so it is 0 in the bulk main run.
INDIC_TIERS: list[Tier] = [
    Tier("A - Verified native",   0.06,  40.0, "100% reserved for anneal + post-train"),
    Tier("B - Unverified crawl",  0.47, 150.0, "bulk of main-run Indic; repeat ~1.1x"),
    Tier("C - Translated",        0.21,  60.0, "fills STEM-in-Indic gaps; repeat ~1.2x"),
    Tier("D - Synthetic",         0.26,   0.0, "generated for uncovered language x topic"),
]

# §5 — reasoning effort bands (shares of the reasoning budget)
REASONING_BANDS: list[Band] = [
    Band("Low",    0.15, "What is 12 x 8? -> direct answer"),
    Band("Medium", 0.30, "single-variable algebra word problem"),
    Band("High",   0.35, "AIME-style combinatorics, multi-path"),
    Band("Ultra",  0.20, "GPQA cross-domain, long self-correcting"),
]

# §6 — anneal mixture (shares of ANNEAL_B). reserve_supply_b set where the slot
# is fed from a deliberately-withheld scarce reserve.
ANNEAL_LANES: list[AnnealLane] = [
    AnnealLane("General web",  0.10,               concentrates="sharply reduced"),
    AnnealLane("Code",         0.20,               concentrates="hardest / test-passing patches"),
    AnnealLane("Indic",        0.20, 40.0,         concentrates="100% Tier A verified native"),
    AnnealLane("STEM/math",    0.15,               concentrates="hardest tier only"),
    AnnealLane("Reasoning",    0.15, 0.20 * 114.0, concentrates="100% ultra band"),
    AnnealLane("Agentic",      0.15, 0.63, reserve_only=False,
               concentrates="real (oversampled ~5x) + best synthetic"),
    AnnealLane("Long-context", 0.05,               concentrates="held flat"),
]


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------
def indic_total_b() -> float:
    """Total Indic across the whole run = main Indic + anneal Indic."""
    main_indic = next(l for l in MAIN_LANES if l.name == "Indic").tokens()
    anneal_indic = next(a for a in ANNEAL_LANES if a.name == "Indic").tokens()
    return main_indic + anneal_indic


def reasoning_total_b() -> float:
    return next(l for l in MAIN_LANES if l.name == "Reasoning traces").tokens()


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def approx(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    # 1. main shares sum to 1.0
    s = sum(l.share for l in MAIN_LANES)
    checks.append(Check("main shares sum to 100%", approx(s, 1.0), f"sum={s:.4f}"))

    # 2. main tokens sum to MAIN_B
    t = sum(l.tokens() for l in MAIN_LANES)
    checks.append(Check("main tokens sum to 1900B", approx(t, MAIN_B, 1e-3), f"sum={t:.1f}B"))

    # 3. protected floors respected
    for l in MAIN_LANES:
        if l.floor:
            checks.append(Check(
                f"{l.name} >= floor {l.floor:.0%}",
                l.share >= l.floor - 1e-9,
                f"share={l.share:.0%}",
            ))

    # 4. indic tier shares sum to 1.0
    s = sum(t.share for t in INDIC_TIERS)
    checks.append(Check("indic tiers sum to 100%", approx(s, 1.0), f"sum={s:.4f}"))

    # 5. indic tier tokens sum to total indic budget
    total_indic = indic_total_b()
    tier_tokens = sum(t.share * total_indic for t in INDIC_TIERS)
    checks.append(Check(
        "indic tier tokens == total indic budget",
        approx(tier_tokens, total_indic, 1e-3),
        f"tiers={tier_tokens:.1f}B vs budget={total_indic:.1f}B",
    ))

    # 6. reasoning bands sum to 1.0
    s = sum(b.share for b in REASONING_BANDS)
    checks.append(Check("reasoning bands sum to 100%", approx(s, 1.0), f"sum={s:.4f}"))

    # 7. anneal shares sum to 1.0
    s = sum(a.share for a in ANNEAL_LANES)
    checks.append(Check("anneal shares sum to 100%", approx(s, 1.0), f"sum={s:.4f}"))

    # 8. reserve feasibility for every reserve-fed anneal slot
    for a in ANNEAL_LANES:
        if a.reserve_supply_b is None:
            continue
        if a.reserve_only:
            oversample = a.tokens() / a.reserve_supply_b if a.reserve_supply_b else float("inf")
            ok = oversample <= MAX_OVERSAMPLE
            checks.append(Check(
                f"anneal '{a.name}' reserve fills slot within {MAX_OVERSAMPLE:.0f}x",
                ok,
                f"slot={a.tokens():.1f}B / reserve={a.reserve_supply_b:.2f}B "
                f"= {oversample:.1f}x oversample"
                + ("" if ok else "  -> INFEASIBLE from reserve alone"),
            ))
        else:
            # reserve oversampled at real_oversample, synthetic covers the rest
            topup = a.synthetic_topup_b()
            checks.append(Check(
                f"anneal '{a.name}' feasible (reserve {a.real_oversample:.0f}x + synthetic)",
                topup >= 0.0,
                f"slot={a.tokens():.1f}B = real {a.reserve_supply_b * a.real_oversample:.1f}B "
                f"({a.real_oversample:.0f}x of {a.reserve_supply_b:.2f}B) + synthetic {topup:.1f}B",
            ))

    # 9. Tier A reserve is not overdrawn by the anneal Indic slot
    tier_a = next(t for t in INDIC_TIERS if t.name.startswith("A"))
    anneal_indic = next(a for a in ANNEAL_LANES if a.name == "Indic")
    checks.append(Check(
        "Tier-A supply covers anneal Indic slot",
        tier_a.supply_b >= anneal_indic.tokens() - 1e-9,
        f"supplyA={tier_a.supply_b:.0f}B >= slot={anneal_indic.tokens():.0f}B",
    ))

    return checks


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _verdict(l: Lane) -> str:
    r = l.repetition_factor()
    if r == float("inf"):
        return "no real supply"
    if r <= 0.5:
        return f"oversupplied ~{1/r:.1f}x"
    if r < 0.98:
        return f"covered ({1/r:.1f}x headroom)"
    if r <= 1.02:
        return "at supply, no margin"
    if r <= 2.0:
        return f"short -> repeat ~{r:.2f}x"
    return f"scarce -> ~{(1 - l.supply_b / l.tokens()) * 100:.0f}% synthetic"


def print_report() -> None:
    print("=" * 78)
    print("V5 DATA MIXTURE — computed report")
    print(f"budget: {TOTAL_B:.0f}B total = {MAIN_B:.0f}B main + {ANNEAL_B:.0f}B anneal")
    print("=" * 78)

    print("\n§2  MAIN PRETRAINING MIXTURE (of 1900B)")
    print(f"{'Lane':<18}{'Share':>7}{'Tokens':>9}{'Supply':>9}  Verdict")
    print("-" * 78)
    for l in MAIN_LANES:
        print(f"{l.name:<18}{l.share:>6.0%}{l.tokens():>8.0f}B{l.supply_b:>8.0f}B  {_verdict(l)}")
    print("-" * 78)
    print(f"{'TOTAL':<18}{sum(l.share for l in MAIN_LANES):>6.0%}"
          f"{sum(l.tokens() for l in MAIN_LANES):>8.0f}B")

    total_indic = indic_total_b()
    print(f"\n§3  INDIC TIER SPLIT (of {total_indic:.0f}B total Indic)")
    print(f"{'Tier':<24}{'Share':>7}{'Tokens':>9}{'Supply':>9}  Note")
    print("-" * 78)
    for t in INDIC_TIERS:
        tok = t.share * total_indic
        print(f"{t.name:<24}{t.share:>6.0%}{tok:>8.0f}B{t.supply_b:>8.0f}B  {t.note}")

    rb = reasoning_total_b()
    print(f"\n§5  REASONING EFFORT BANDS (of {rb:.0f}B)")
    print(f"{'Band':<10}{'Share':>7}{'Tokens':>9}  Example")
    print("-" * 78)
    for b in REASONING_BANDS:
        print(f"{b.name:<10}{b.share:>6.0%}{b.share * rb:>8.1f}B  {b.example}")

    print(f"\n§6  ANNEAL MIXTURE (of {ANNEAL_B:.0f}B)")
    print(f"{'Lane':<14}{'Share':>7}{'Tokens':>9}  Concentrates")
    print("-" * 78)
    for a in ANNEAL_LANES:
        print(f"{a.name:<14}{a.share:>6.0%}{a.tokens():>8.1f}B  {a.concentrates}")

    print("\nINVARIANT CHECKS")
    print("-" * 78)
    checks = run_checks()
    for c in checks:
        mark = "PASS" if c.ok else "FAIL"
        print(f"[{mark}] {c.name:<48} {c.detail}")
    failed = [c for c in checks if not c.ok]
    print("-" * 78)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILURES:")
        for c in failed:
            print(f"  - {c.name}: {c.detail}")


def main(argv: list[str]) -> int:
    checks = run_checks()
    failed = [c for c in checks if not c.ok]
    if "--check" in argv:
        for c in failed:
            print(f"FAIL: {c.name}: {c.detail}")
        if not failed:
            print(f"OK: all {len(checks)} invariants hold")
        return 1 if failed else 0
    print_report()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
