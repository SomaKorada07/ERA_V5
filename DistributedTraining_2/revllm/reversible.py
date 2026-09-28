"""Memory-free reversible transformer stack (leapfrog / midpoint integrator).

The rule, from Gal, Eliasof, Turek, Ascher, Treister and Haber, *Reversing Large
Language Models for Efficient Training and Fine-Tuning* (Nov 2025):

    p_{l+1} = p_{l-1} + 2h * f_theta_l(p_l)               (forward)
    p_{l-1} = p_{l+1} - 2h * f_theta_l(p_l)               (reverse)

Because layer ``l`` is evaluated at ``p_l`` — the state the backward pass already
holds — the update can be run in both directions. The forward pass keeps only the
two boundary states; the backward pass walks *down* the stack, reconstructing each
input from the output it already has, and never stores an intermediate activation.
Activation memory is therefore independent of depth.

Two carried gradients ``(a, b) = (dL/dp_{l+1}, dL/dp_l)`` are propagated with the
leapfrog adjoint recurrence::

    Js, *Jp = grad(f(p_l), [p_l, *params], grad_outputs=a)
    a, b    = (b + 2h * Js,  a)          # new (dL/dp_l, dL/dp_{l-1})
    grad_params_l += 2h * Jp

Constraints (both enforced/assumed here):
  * the forward must be deterministic — dropout is 0 in reversible mode;
  * the rule is only marginally stable — the step size ``h`` must be kept small.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _flat_params(blocks: nn.ModuleList) -> List[torch.Tensor]:
    return [p for blk in blocks for p in blk.parameters()]


class _ReversibleStack(torch.autograd.Function):
    """Leapfrog reversible stack with a reconstruction-based backward pass."""

    @staticmethod
    def forward(ctx, x, step_size, blocks, n_params_per_block, *params):
        ctx.blocks = blocks
        ctx.step_size = step_size
        ctx.n_params_per_block = n_params_per_block

        h2 = 2.0 * step_size
        with torch.no_grad():
            p_prev = x                       # p_{-1}  (== p_0)
            p_cur = x                        # p_0
            for blk in blocks:
                p_next = p_prev + h2 * blk(p_cur)
                p_prev, p_cur = p_cur, p_next
        # keep only the two boundary states
        ctx.save_for_backward(p_cur.detach(), p_prev.detach(), *params)
        return p_cur

    @staticmethod
    def backward(ctx, grad_output):
        blocks = ctx.blocks
        h2 = 2.0 * ctx.step_size
        saved = ctx.saved_tensors
        s1 = saved[0].clone()                # p_L
        s0 = saved[1].clone()                # p_{L-1}
        params = list(saved[2:])

        a = grad_output                      # dL/dp_L
        b = torch.zeros_like(grad_output)    # dL/dp_{L-1} (not an output)
        param_grads = [torch.zeros_like(p) for p in params]

        n = ctx.n_params_per_block
        for l in range(len(blocks) - 1, -1, -1):
            blk = blocks[l]
            blk_params = [p for p in blk.parameters()]

            # recompute f(p_l) with grad enabled at the held state s0 = p_l
            s0_ = s0.detach().requires_grad_(True)
            with torch.enable_grad():
                y = blk(s0_)
            grads = torch.autograd.grad(
                y, [s0_, *blk_params], grad_outputs=a, retain_graph=False
            )
            js, jp = grads[0], grads[1:]

            # leapfrog adjoint: (a, b) <- (b + 2h*Js, a)
            a, b = b + h2 * js, a
            # accumulate parameter grads for this block: 2h * Jp
            base = l * n
            for i, g in enumerate(jp):
                param_grads[base + i] = h2 * g

            # reconstruct p_{l-1} = p_{l+1} - 2h*f(p_l);  shift states down
            with torch.no_grad():
                p_prev = s1 - h2 * y.detach()
            s1, s0 = s0, p_prev

        # p_{-1} == p_0 == x, so both boundary grads flow into the input
        grad_x = a + b
        return (grad_x, None, None, None, *param_grads)


def reversible_stack(x: torch.Tensor, blocks: nn.ModuleList, step_size: float) -> torch.Tensor:
    """Run ``blocks`` as a memory-free reversible leapfrog stack."""
    n_params_per_block = len(list(blocks[0].parameters()))
    params = _flat_params(blocks)
    return _ReversibleStack.apply(x, step_size, blocks, n_params_per_block, *params)


# --------------------------------------------------------------------------- #
# Reference implementations used only for correctness / stability checks.
# --------------------------------------------------------------------------- #
def leapfrog_reference(x, blocks, step_size):
    """Same leapfrog recurrence but with ordinary autograd (stores activations).

    Used in tests to check that the memory-free custom backward produces identical
    gradients.
    """
    h2 = 2.0 * step_size
    p_prev, p_cur = x, x
    for blk in blocks:
        p_next = p_prev + h2 * blk(p_cur)
        p_prev, p_cur = p_cur, p_next
    return p_cur


@torch.no_grad()
def reconstruction_error(x, blocks, step_size, integrator="leapfrog"):
    """How well can the input be rebuilt from the output by running in reverse?

    Returns (relative L2 error of the reconstructed input, output tensor).

    * ``leapfrog`` reverses exactly (error ~ machine precision) because layer l is
      evaluated at the *held* state p_l.
    * ``euler`` (``p_{l+1} = p_l + h f(p_l)``) cannot be reversed: recovering p_l
      would need f(p_l), which depends on the unknown p_l. Reversing with the wrong
      evaluation point accumulates error — this is why the naive variant fails.
    """
    h = step_size
    h2 = 2.0 * step_size
    if integrator == "leapfrog":
        # forward, keep boundary states
        p_prev, p_cur = x, x
        for blk in blocks:
            p_next = p_prev + h2 * blk(p_cur)
            p_prev, p_cur = p_cur, p_next
        s1, s0 = p_cur, p_prev
        # reverse
        for l in range(len(blocks) - 1, -1, -1):
            p_back = s1 - h2 * blocks[l](s0)
            s1, s0 = s0, p_back
        rebuilt = s0  # p_{-1} == x
    elif integrator == "euler":
        acts = [x]
        p = x
        for blk in blocks:
            p = p + h * blk(p)
            acts.append(p)
        out = p
        # naive reverse: evaluate the block at the *output* (wrong point)
        p = out
        for l in range(len(blocks) - 1, -1, -1):
            p = p - h * blocks[l](p)
        rebuilt = p
        p_cur = out
    else:
        raise ValueError(integrator)
    err = (rebuilt - x).norm() / (x.norm() + 1e-12)
    return err.item(), p_cur
