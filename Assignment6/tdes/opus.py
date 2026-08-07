"""OPUS data selector: scores candidate batches, accepts/rejects/defers, and rescues
protected lanes the (deliberately biased) proxy would otherwise starve.

The proxy is intentionally English-heavy: it under-scores Indic / agentic / reasoning
batches, so without protected-floor overrides those lanes collapse toward zero. Every
decision is recorded so the stream is auditable (Session 5 §7, §10).
"""

from __future__ import annotations

import hashlib

from .util import Config, PROTECTED_LANES

# proxy bias: positive = proxy over-values, negative = proxy under-values
_PROXY_BIAS = {
    "general_web": +0.15,
    "code": +0.10,
    "math_science": +0.05,
    "indic": -0.15,
    "agentic": -0.20,
    "reasoning": -0.10,
}


def _score(candidate_key: str, lane: str, cfg: Config) -> float:
    h = hashlib.sha256((candidate_key + cfg.proxy_version).encode()).hexdigest()
    raw = int(h[:12], 16) / float(16 ** 12)          # deterministic uniform [0,1)
    biased = raw + _PROXY_BIAS.get(lane, 0.0)
    return max(0.0, min(1.0, biased))


def opus_decide(candidate_key: str, lane: str, stage: str, checkpoint_step: int,
                loss_tokens: int, cfg: Config) -> dict:
    """Return a full candidate decision record."""
    threshold = 1.0 - cfg.opus_accept_fraction
    defer_low = threshold - cfg.opus_defer_band
    score = _score(candidate_key, lane, cfg)

    protected = lane in PROTECTED_LANES
    override = False
    if score >= threshold:
        status, reason = "accepted", "score>=threshold"
    elif protected:
        status, reason, override = "accepted", "protected_floor_override", True
    elif score >= defer_low:
        status, reason = "deferred", "near_threshold_band"
    else:
        status, reason = "rejected", "low_proxy_utility"

    # effective-token estimate: value-weighted loss tokens if accepted
    eff = round(loss_tokens * (0.5 + score), 2) if status == "accepted" else 0.0

    return {
        "candidate_id": candidate_key,
        "capability_lane": lane,
        "curriculum_stage": stage,
        "checkpoint_step_for_scoring": checkpoint_step,
        "proxy_version": cfg.proxy_version,
        "opus_score": round(score, 6),
        "threshold": round(threshold, 4),
        "status": status,
        "reason": reason,
        "protected_floor_override": override,
        "effective_token_estimate": eff,
    }
