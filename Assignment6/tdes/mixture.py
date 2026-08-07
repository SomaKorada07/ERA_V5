"""Mixture timeline compiler: curriculum stages -> executable per-step lane quotas.

Converts the Session 5 human-readable plan (stages, lane weights, protected floors,
anneal reserves) into a concrete per-stage batch/lane schedule, and flags lanes that
cannot be satisfied from available shards (requiring repetition or synthetic data).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .util import CAPABILITY_LANES, PROTECTED_LANES, Config


@dataclass
class Stage:
    name: str
    n_global_batches: int          # how many optimizer steps this stage plans
    mixture: dict[str, float]      # lane -> target share (sums ~1.0)
    protected_floors: dict[str, float]
    warmup_batches: int = 1
    anneal_reserve: bool = False   # if True, draw scarce reserves (Tier-A / real)


def default_stages() -> list[Stage]:
    """A compact 4-stage curriculum mirroring Session 5 (general -> code/STEM ->
    long/reasoning -> anneal)."""
    return [
        Stage(
            "general", n_global_batches=8,
            mixture={"general_web": 0.55, "code": 0.15, "math_science": 0.10,
                     "indic": 0.12, "reasoning": 0.05, "agentic": 0.03},
            protected_floors={"indic": 0.12, "agentic": 0.02, "reasoning": 0.05},
            warmup_batches=2,
        ),
        Stage(
            "code_stem_ramp", n_global_batches=10,
            mixture={"general_web": 0.30, "code": 0.28, "math_science": 0.18,
                     "indic": 0.12, "reasoning": 0.08, "agentic": 0.04},
            protected_floors={"indic": 0.12, "agentic": 0.02, "reasoning": 0.05},
            warmup_batches=2,
        ),
        Stage(
            "reasoning_midtrain", n_global_batches=8,
            mixture={"general_web": 0.22, "code": 0.22, "math_science": 0.18,
                     "indic": 0.14, "reasoning": 0.16, "agentic": 0.08},
            protected_floors={"indic": 0.12, "agentic": 0.05, "reasoning": 0.10},
            warmup_batches=1,
        ),
        Stage(
            "anneal", n_global_batches=6,
            mixture={"general_web": 0.10, "code": 0.20, "math_science": 0.15,
                     "indic": 0.20, "reasoning": 0.20, "agentic": 0.15},
            protected_floors={"indic": 0.15, "agentic": 0.10, "reasoning": 0.15},
            warmup_batches=1, anneal_reserve=True,
        ),
    ]


def validate_stage(stage: Stage) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    s = sum(stage.mixture.values())
    if abs(s - 1.0) > 1e-6:
        problems.append(f"stage '{stage.name}' mixture sums to {s:.3f}, not 1.0")
    for lane, floor in stage.protected_floors.items():
        if stage.mixture.get(lane, 0.0) < floor - 1e-9:
            problems.append(
                f"stage '{stage.name}' lane '{lane}' share "
                f"{stage.mixture.get(lane, 0):.3f} < floor {floor:.3f}")
    for lane in stage.mixture:
        if lane not in CAPABILITY_LANES:
            problems.append(f"stage '{stage.name}' unknown lane '{lane}'")
    return problems


def lane_plan_for_stage(stage: Stage, seqs_per_batch: int) -> list[list[str]]:
    """Deterministically assign a lane to every sequence of every global batch in
    a stage so that realized shares approximate the target mixture, while never
    dropping a protected lane below its floor.

    Uses the largest-remainder method per batch for stability & reproducibility.
    """
    plans: list[list[str]] = []
    carry = {lane: 0.0 for lane in stage.mixture}
    for _ in range(stage.n_global_batches):
        # desired count per lane this batch = share * seqs, distributed by
        # largest remainder, with protected floors enforced first.
        counts = {lane: 0 for lane in stage.mixture}
        remaining = seqs_per_batch

        # 1) protected floors get their guaranteed minimum (>=1 if floor>0)
        for lane, floor in sorted(stage.protected_floors.items()):
            need = max(1, round(floor * seqs_per_batch)) if floor > 0 else 0
            need = min(need, remaining)
            counts[lane] += need
            remaining -= need

        # 2) distribute the rest by target share using accumulated carry
        if remaining > 0:
            desired = {}
            for lane, share in stage.mixture.items():
                target = share * seqs_per_batch + carry[lane]
                desired[lane] = target
            # allocate floors of each desired, then largest remainder
            base = {lane: int(desired[lane]) for lane in desired}
            # do not double-count protected minimums already given
            for lane in base:
                base[lane] = max(0, base[lane] - counts[lane])
            alloc = min(remaining, sum(base.values()))
            # assign base up to remaining
            order = sorted(base, key=lambda l: -base[l])
            for lane in order:
                if remaining <= 0:
                    break
                take = min(base[lane], remaining)
                counts[lane] += take
                remaining -= take
            # any leftover -> largest fractional remainder
            fracs = sorted(desired, key=lambda l: -(desired[l] - int(desired[l])))
            i = 0
            while remaining > 0 and fracs:
                counts[fracs[i % len(fracs)]] += 1
                remaining -= 1
                i += 1

        # update carry (target minus realized) so long-run shares converge
        for lane, share in stage.mixture.items():
            carry[lane] += share * seqs_per_batch - counts[lane]

        # expand counts into an ordered lane list (sorted for determinism)
        lanes: list[str] = []
        for lane in sorted(counts):
            lanes.extend([lane] * counts[lane])
        plans.append(lanes)
    return plans


def compile_timeline(cfg: Config, stages: list[Stage],
                     lane_supply_tokens: dict[str, int]) -> dict:
    """Produce an executable timeline + a scarcity report."""
    seqs = cfg.global_batch_seqs
    timeline = []
    token_cursor = 0
    scarcity = {}

    # estimate demand per lane across the whole run
    demand = {lane: 0 for lane in CAPABILITY_LANES}
    for stage in stages:
        plan = lane_plan_for_stage(stage, seqs)
        for lanes in plan:
            for lane in lanes:
                demand[lane] += cfg.seq_len

    for lane in CAPABILITY_LANES:
        supply = lane_supply_tokens.get(lane, 0)
        need = demand[lane]
        if supply <= 0:
            scarcity[lane] = {"demand": need, "supply": supply,
                              "action": "synthesize", "repetition_factor": None}
        elif need > supply:
            scarcity[lane] = {"demand": need, "supply": supply,
                              "action": "repeat",
                              "repetition_factor": round(need / supply, 3)}
        else:
            scarcity[lane] = {"demand": need, "supply": supply,
                              "action": "covered",
                              "repetition_factor": round(need / supply, 3)}

    for stage in stages:
        plan = lane_plan_for_stage(stage, seqs)
        stage_tokens = stage.n_global_batches * cfg.global_batch_tokens
        timeline.append({
            "stage": stage.name,
            "token_start": token_cursor,
            "token_end": token_cursor + stage_tokens,
            "sequence_length": cfg.seq_len,
            "mixture": stage.mixture,
            "protected_floors": stage.protected_floors,
            "warmup_batches": stage.warmup_batches,
            "anneal_reserve": stage.anneal_reserve,
            "n_global_batches": stage.n_global_batches,
            "lane_plan": plan,
        })
        token_cursor += stage_tokens

    return {"timeline": timeline, "scarcity": scarcity, "total_tokens": token_cursor}
