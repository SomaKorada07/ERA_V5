"""
Proxy-experiment decision logic (README §9).

The whole point of §9 is that every mixture number is a *hypothesis* with a
pre-committed pass/fail rule. This module encodes those rules as pure functions
so the decision is mechanical, not a judgement call made after seeing results.

Feed in the metrics a proxy run would produce; get back the committed decision.
Running the file executes worked examples for each experiment.

    python3 proxy_experiments.py

No third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    verdict: str          # short machine-readable outcome
    action: str           # what we do to the mixture as a result

    def __str__(self) -> str:
        return f"{self.verdict}  ->  {self.action}"


# --------------------------------------------------------------------------
# Experiment 1 — is the 17% Indic share worth its cost?  (1B params, 20B tokens)
# --------------------------------------------------------------------------
def experiment_1(
    milu_12: float,
    milu_17: float,
    milu_25: float,
    milu_d_full: float,
    milu_d_cut: float,
) -> Decision:
    """
    milu_12/17/25 : MILU accuracy at Indic share = 12% (floor) / 17% / 25%.
    milu_d_full   : MILU with the full 26% synthetic Tier-D.
    milu_d_cut    : MILU with Tier-D cut 26% -> 10%.

    Committed rules:
      - 12%-floor within 2 pts of 17%  -> the extra 5 pts buys nothing;
        drop toward the floor, redirect freed budget to Code.
      - 25% beats 17% by > 3 pts AND synthetic quality holds -> raise Indic.
      - Tier-D cut drops MILU < 1 pt -> D is overweight, trim it.
      - otherwise -> 17% / current tier split stands.
    """
    synthetic_helps = (milu_d_full - milu_d_cut) >= 1.0

    if (milu_17 - milu_12) <= 2.0:
        return Decision(
            "FLOOR_SUFFICIENT",
            "drop Indic 17% -> 12% floor; redirect ~95B to Code",
        )
    if (milu_25 - milu_17) > 3.0 and synthetic_helps:
        return Decision(
            "RAISE_INDIC",
            "raise Indic 17% -> 25%; verify Tier-D quality does not regress",
        )
    if not synthetic_helps:
        return Decision(
            "TRIM_SYNTHETIC",
            "Tier-D overweight; cut D 26% -> ~10%, backfill with B repetition",
        )
    return Decision("HOLD_17", "keep Indic at 17% with the 6/47/21/26 tier split")


# --------------------------------------------------------------------------
# Experiment 2 — synthetic agentic quality + real-data reserve  (3B, 60B)
# --------------------------------------------------------------------------
def experiment_2(tau_synth_only: float, tau_blend: float) -> Decision:
    """
    tau_synth_only : tau-bench completion, 100% synthetic agentic data.
    tau_blend      : tau-bench completion, synthetic + 0.63B real oversampled 4x.

    Committed rule:
      - blend beats synth-only by > 5 pts -> reserving real trajectories for the
        anneal is high value; keep the §4/§6 reservation.
      - otherwise -> scarcity value overstated; real data can be spent earlier.
    """
    if (tau_blend - tau_synth_only) > 5.0:
        return Decision(
            "RESERVE_CONFIRMED",
            "keep 0.63B real trajectories reserved for the anneal (§4/§6)",
        )
    return Decision(
        "RESERVE_NOT_WORTH_IT",
        "spend real trajectories in the main run; reserve value overstated",
    )


# --------------------------------------------------------------------------
# Experiment 3 — staged curriculum vs. flat mixture  (1B, 20B)
# --------------------------------------------------------------------------
def experiment_3(
    staged_aggregate: float,
    flat_aggregate: float,
    staged_grad_spike: float,
    flat_grad_spike: float,
) -> Decision:
    """
    *_aggregate   : summed MMLU + LiveCodeBench + RULER.
    *_grad_spike  : peak gradient-norm multiplier at mixture transitions.

    Committed rule:
      - staged beats flat by > 1.5 pts aggregate OR is materially more stable
        (lower grad spike) -> keep the staged curriculum (§8).
      - otherwise -> ordering complexity isn't paying for itself; simplify.
    """
    quality_win = (staged_aggregate - flat_aggregate) > 1.5
    stability_win = staged_grad_spike < flat_grad_spike * 0.75
    if quality_win or stability_win:
        return Decision(
            "KEEP_CURRICULUM",
            "retain staged general->code/STEM->long-context->anneal ordering",
        )
    return Decision(
        "SIMPLIFY",
        "curriculum not worth the complexity; flatten toward a single mixture",
    )


# --------------------------------------------------------------------------
# Worked examples (illustrative inputs — NOT real run results)
# --------------------------------------------------------------------------
def _demo() -> None:
    print("Proxy-experiment decision logic — worked examples")
    print("(inputs are illustrative placeholders, not real proxy-run numbers)\n")

    print("Experiment 1  (1B / 20B) — Indic share")
    for label, args in [
        ("17% clearly helps, D helps", (41.0, 45.0, 46.5, 45.0, 43.0)),
        ("floor is good enough",        (44.0, 45.0, 45.5, 45.0, 44.5)),
        ("25% wins big, D helps",       (41.0, 45.0, 49.0, 45.0, 43.0)),
        ("synthetic D is dead weight",  (41.0, 45.0, 46.0, 45.0, 44.6)),
    ]:
        print(f"  {label:<28} {experiment_1(*args)}")

    print("\nExperiment 2  (3B / 60B) — agentic reserve")
    for label, args in [
        ("real reserve pays off", (38.0, 46.0)),
        ("reserve not worth it",  (44.0, 45.5)),
    ]:
        print(f"  {label:<28} {experiment_2(*args)}")

    print("\nExperiment 3  (1B / 20B) — curriculum ordering")
    for label, args in [
        ("staged wins on quality",   (182.0, 179.0, 3.0, 4.0)),
        ("staged wins on stability", (180.5, 180.0, 1.5, 6.0)),
        ("no benefit -> simplify",   (180.2, 180.0, 3.9, 4.0)),
    ]:
        print(f"  {label:<28} {experiment_3(*args)}")


if __name__ == "__main__":
    _demo()
