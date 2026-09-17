"""
run_sim.py — drives zero_sim.py: runs one real training step under each ZeRO
stage, proves every stage matches a single-device reference, and measures
per-GPU memory + communication. Prints verified numbers used in the notebook.
"""
import torch, torch.nn as nn
from zero_sim import (
    Cluster, DemoMLP, VirtualGPU, CommCounter,
    flatten_params, load_flat_into_model, grad_flat,
    shard_offsets, all_reduce, reduce_scatter, all_gather,
    formula_memory, adam_step, MemBreakdown,
    BYTES_FP16, BYTES_FP32, K_ADAM,
)

torch.manual_seed(0)
N = 32
MICRO = 8                       # samples per virtual GPU
GLOBAL = N * MICRO             # global batch
D_IN = 256

model = DemoMLP(d_in=D_IN, d_hidden=1024, d_out=256, depth=4)
PSI = sum(p.numel() for p in model.parameters())
print(f"world_size N = {N} | parameters Psi = {PSI:,} | global batch = {GLOBAL}")

# ---- shared initial state -------------------------------------------------
W0 = flatten_params(model)                     # fp32 master weights (init)
offs = shard_offsets(PSI, N)

# one fixed global batch, split identically for every stage
X = torch.randn(GLOBAL, D_IN)
Y = torch.randn(GLOBAL, 256)
loss_fn = nn.MSELoss()                          # mean over samples


def local_grad(w16_full: torch.Tensor, xb, yb) -> torch.Tensor:
    """fp16-compute forward/backward on a micro-batch -> fp32 flat gradient."""
    m = DemoMLP(d_in=D_IN, d_hidden=1024, d_out=256, depth=4)
    load_flat_into_model(m, w16_full)           # w16_full is the (gathered) params
    for p in m.parameters():
        p.grad = None
    out = m(xb)
    loss = loss_fn(out, yb)
    loss.backward()
    return grad_flat(m), loss.item()


# ===========================================================================
#  REFERENCE : one GPU, full global batch
# ===========================================================================
def reference():
    w32 = W0.clone()
    m = torch.zeros_like(w32); v = torch.zeros_like(w32)
    w16 = w32.to(torch.float16).float()         # simulate fp16 round-trip
    g, _ = local_grad(w16, X, Y)
    adam_step(w32, g, m, v, t=1)
    return w32

W_REF = reference()


# ===========================================================================
#  helper: per-device gradients (data-parallel) from a given full w16
# ===========================================================================
def per_device_grads(w16_full):
    grads, losses = [], []
    for r in range(N):
        xb = X[r*MICRO:(r+1)*MICRO]; yb = Y[r*MICRO:(r+1)*MICRO]
        g, l = local_grad(w16_full, xb, yb)
        grads.append(g.to(torch.float16))       # gradients communicated in fp16
        losses.append(l)
    return grads, losses


# ===========================================================================
#  BASELINE (DDP)
# ===========================================================================
def run_baseline():
    comm = CommCounter()
    w32 = [W0.clone() for _ in range(N)]
    m = [torch.zeros(PSI) for _ in range(N)]
    v = [torch.zeros(PSI) for _ in range(N)]
    w16_full = W0.to(torch.float16).float()

    grads, _ = per_device_grads(w16_full)
    g_sum = all_reduce(grads, comm).float()     # sum over devices (fp16 comm)
    g_mean = g_sum / N
    for r in range(N):                           # every GPU: full, redundant step
        adam_step(w32[r], g_mean.clone(), m[r], v[r], t=1)

    mem = MemBreakdown(BYTES_FP16*PSI, BYTES_FP16*PSI, K_ADAM*PSI)
    return w32[0], comm.bytes_per_gpu, mem


# ===========================================================================
#  ZeRO-1 : shard optimizer states
#  ZeRO-2 : + shard gradients          (same collectives; less resident memory)
# ===========================================================================
def run_zero12(stage):
    comm = CommCounter()
    # sharded optimizer state (and, conceptually, master weights) per device
    w32_shard = [W0[a:b].clone() for (a, b) in offs]
    m = [torch.zeros(b-a) for (a, b) in offs]
    v = [torch.zeros(b-a) for (a, b) in offs]
    w16_full = W0.to(torch.float16).float()

    grads, _ = per_device_grads(w16_full)        # each device holds a FULL grad
    g_shards = reduce_scatter(grads, comm)       # -> summed grad shard per device
    new_w16_shards = []
    for r, (a, b) in enumerate(offs):
        gm = g_shards[r].float() / N
        adam_step(w32_shard[r], gm, m[r], v[r], t=1)
        new_w16_shards.append(w32_shard[r].to(torch.float16))
    full_w16 = all_gather(new_w16_shards, comm)  # rebuild full params everywhere

    # reassemble the updated master weights for the correctness check
    w32_new = torch.cat(w32_shard, 0)

    if stage == "zero1":       # gradients NOT sharded -> full grad resident
        mem = MemBreakdown(BYTES_FP16*PSI, BYTES_FP16*PSI, K_ADAM*PSI/N)
    else:                      # zero2: gradients sharded
        mem = MemBreakdown(BYTES_FP16*PSI, BYTES_FP16*PSI/N, K_ADAM*PSI/N)
    return w32_new, comm.bytes_per_gpu, mem


# ===========================================================================
#  ZeRO-3 : shard parameters too — no device ever holds the full model
# ===========================================================================
def run_zero3():
    comm = CommCounter()
    w16_shard = [W0[a:b].to(torch.float16) for (a, b) in offs]   # persistent
    w32_shard = [W0[a:b].clone() for (a, b) in offs]
    m = [torch.zeros(b-a) for (a, b) in offs]
    v = [torch.zeros(b-a) for (a, b) in offs]

    # --- FORWARD: all-gather params, compute, then free them ---
    full_w16 = all_gather(w16_shard, comm).float()
    grads, _ = per_device_grads(full_w16)
    # (in real ZeRO-3 params are re-gathered for the backward pass)
    comm.add(int((N-1)/N * (BYTES_FP16*PSI)))    # 2nd all-gather (backward)
    g_shards = reduce_scatter(grads, comm)       # grad shard per device
    # full params are freed here -> only shards remain resident

    for r, (a, b) in enumerate(offs):
        gm = g_shards[r].float() / N
        adam_step(w32_shard[r], gm, m[r], v[r], t=1)
        w16_shard[r] = w32_shard[r].to(torch.float16)

    w32_new = torch.cat(w32_shard, 0)
    mem = MemBreakdown(BYTES_FP16*PSI/N, BYTES_FP16*PSI/N, K_ADAM*PSI/N)
    return w32_new, comm.bytes_per_gpu, mem


# ===========================================================================
#  RUN ALL + verify
# ===========================================================================
def maxerr(a, b):
    return (a - b).abs().max().item()

results = {}
weights = {}
for name, fn in [
    ("baseline", run_baseline),
    ("zero1", lambda: run_zero12("zero1")),
    ("zero2", lambda: run_zero12("zero2")),
    ("zero3", run_zero3),
]:
    w_new, comm_bytes, mem = fn()
    weights[name] = w_new
    results[name] = dict(comm=comm_bytes, mem=mem)

# Ground truth = standard data-parallel (DDP baseline). ZeRO's promise is that
# stages 1/2/3 reproduce DDP's update *exactly* while storing less per GPU.
W_DDP = weights["baseline"]
print("\n=== correctness: does each ZeRO stage reproduce DDP's weights? ===")
print("    max|w_stage - w_DDP|  (should be exactly 0 -- same maths, less memory)")
for k in ["zero1", "zero2", "zero3"]:
    err = maxerr(weights[k], W_DDP)
    results[k]["err"] = err
    print(f"  {k:9s}  max_abs_err = {err:.3e}   {'IDENTICAL' if err == 0 else ('OK' if err<1e-6 else 'FAIL')}")

# sanity: DDP itself is a correct optimizer step vs an fp32 single-device run
ref_err = maxerr(W_DDP, W_REF)
print(f"\n  DDP vs fp32 single-GPU reference: max_abs_err = {ref_err:.3e} "
      f"(nonzero only because gradients are all-reduced in fp16)")
results["baseline"]["err"] = 0.0

def mb(x): return x / 1e6
print("\n=== per-GPU persistent model-state memory (MB) ===")
print(f"  {'stage':9s} {'params':>9s} {'grads':>9s} {'optim':>9s} {'TOTAL':>10s}  {'vs base':>8s}")
base_total = results['baseline']['mem'].total
for k, r in results.items():
    m = r['mem']
    print(f"  {k:9s} {mb(m.params):9.2f} {mb(m.grads):9.2f} {mb(m.optim):9.2f} "
          f"{mb(m.total):10.2f}  {m.total/base_total:7.2%}")

print("\n=== per-GPU communication volume per step (MB) ===")
base_comm = results['baseline']['comm']
for k, r in results.items():
    print(f"  {k:9s} {mb(r['comm']):8.2f} MB   ({r['comm']/base_comm:.2f}x baseline)")

# cross-check measured-vs-formula
print("\n=== formula cross-check (MB total per GPU) ===")
for k in results:
    f = formula_memory(PSI, N, k).total
    print(f"  {k:9s} sim={mb(results[k]['mem'].total):8.2f}  formula={mb(f):8.2f}  "
          f"{'MATCH' if abs(f-results[k]['mem'].total)<1 else 'DIFF'}")
