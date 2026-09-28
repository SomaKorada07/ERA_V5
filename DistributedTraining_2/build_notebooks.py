"""Build the three assignment notebooks with nbformat.

Design: heavy training lives in `revllm/` and ran via run_experiments.py; the notebooks
present that work. Each notebook mixes explanation, fast live checks (gradient
correctness, reconstruction, a short demo-train), and the authoritative full 50M-token
numbers loaded from results/*.json. So `nbconvert --execute` stays fast while the
reported figures are the real ones.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PREAMBLE = """\
import os, sys, json
# make the revllm package importable whether run from repo root or notebooks/
REPO = None
for p in [os.getcwd(), os.path.dirname(os.getcwd())]:
    if os.path.exists(os.path.join(p, "revllm")):
        REPO = p
        if p not in sys.path:
            sys.path.insert(0, p)
        break
assert REPO, "could not locate the revllm package"
RESULTS = os.path.join(REPO, "results")
import torch
from revllm.engine import get_device
DEVICE = get_device()
print("repo:", REPO, "| device:", DEVICE, "| torch:", torch.__version__)"""


def md(s):
    return new_markdown_cell(s)


def code(s):
    return new_code_cell(s)


def save(nb, path):
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Notebook 1 — baseline
# --------------------------------------------------------------------------- #
def nb1():
    nb = new_notebook()
    c = nb.cells
    c.append(md(
        "# 01 · Baseline — a 20M LLM on TinyStories (50M tokens)\n\n"
        "**ERA V5 · Session 13 (Reversibility)**, part 1 of 3.\n\n"
        "Train a ~20M-parameter GPT for a 50M-token budget with a **fixed batch size that "
        "runs**. This is the *standard residual* baseline (the Euler update "
        "`p_{l+1} = p_l + f(p_l)`); autograd stores every layer's activations.\n\n"
        "| setting | value |\n|---|---|\n"
        "| params | ~20.1M |\n| layers × width | 9 × 256 |\n| heads | 8 |\n"
        "| context | 512 |\n| vocab | 50257 (GPT-2 BPE) |\n| data | TinyStories |\n"
        "| optimizer | AdamW, cosine LR, warmup |\n| device | Apple M3 Max (MPS) |\n\n"
        "Hardware: Apple **M3 Max**, 68.7 GB unified memory, `torch` MPS backend "
        "(no CUDA). All modules live in [`revllm/`](../revllm)."
    ))
    c.append(code(PREAMBLE))
    c.append(md("## The model\nA compact GPT. The transformer block is exposed as a pure "
                "function `f_theta(p)` (attention then FFN, no residual add of its own) so "
                "the *same* block definition drives both the standard and reversible stacks."))
    c.append(code(
        "from revllm.model import GPT, GPTConfig\n"
        "cfg = GPTConfig(n_layer=9, n_embd=256, n_head=8, block_size=512,\n"
        "                mode='standard', integrator='euler')\n"
        "model = GPT(cfg)\n"
        "print(f'parameters: {model.num_params()/1e6:.2f}M  (non-embedding {model.num_params(True)/1e6:.2f}M)')"
    ))
    c.append(md("## Data\nTinyStories, tokenized with GPT-2 BPE into `data/{train,val}.bin` "
                "(uint16). `prepare()` is idempotent — it reuses the cache."))
    c.append(code(
        "from revllm.data import prepare, TokenData\n"
        "train_bin, val_bin = prepare()\n"
        "td = TokenData(train_bin)\n"
        "print(f'train tokens: {len(td):,}')\n"
        "import tiktoken; enc = tiktoken.get_encoding('gpt2')\n"
        "print(repr(enc.decode(td.data[:60].tolist())))"
    ))
    c.append(md("## A short live demo (≈300K tokens)\nJust to show the loop runs and the loss "
                "falls. The **reported** numbers come from the full 50M-token run below."))
    c.append(code(
        "from revllm.engine import train_run\n"
        "demo = train_run(cfg, name='demo_baseline', batch_size=32, token_budget=300_000,\n"
        "                 train_bin=train_bin, val_bin=val_bin, device=DEVICE, log_every=5)\n"
        "print(demo.summary())"
    ))
    c.append(md("## Full run — 50M tokens, fixed batch = 32\nLoaded from `results/01_baseline.json`, "
                "produced by `run_experiments.py` (each run in its own process for a clean "
                "peak-memory reading)."))
    c.append(code(
        "res = json.load(open(os.path.join(RESULTS, '01_baseline.json')))\n"
        "print(f\"final train loss : {res['final_train_loss']:.4f}\")\n"
        "print(f\"final val   loss : {res['final_val_loss']:.4f}\")\n"
        "print(f\"speed            : {res['tokens_per_sec']:,.0f} tok/s\")\n"
        "print(f\"peak memory      : {res['peak_mem_bytes']/1e9:.2f} GB\")\n"
        "print(f\"batch / tok-step : {res['batch_size']} / {res['tokens_per_step']:,}\")\n"
        "print(f\"steps / tokens   : {res['steps']:,} / {res['tokens_seen']:,}\")\n"
        "print(f\"wall time        : {res['wall_time_s']/60:.1f} min\")"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "h = res['loss_history']\n"
        "plt.figure(figsize=(7,4))\n"
        "plt.plot([x['tokens']/1e6 for x in h], [x['train_loss'] for x in h])\n"
        "plt.xlabel('tokens (millions)'); plt.ylabel('train loss')\n"
        "plt.title('Baseline (standard/euler) — 20M LLM, 50M tokens'); plt.grid(alpha=0.3); plt.show()"
    ))
    c.append(md("➡️ Continue to **02 · Reversibility**, where the same model is trained with the "
                "leapfrog reversible stack and we compare memory and loss at the same batch."))
    return nb


# --------------------------------------------------------------------------- #
# Notebook 2 — reversibility
# --------------------------------------------------------------------------- #
def nb2():
    nb = new_notebook()
    c = nb.cells
    c.append(md(
        "# 02 · Reversibility — leapfrog stack, same batch\n\n"
        "**ERA V5 · Session 13 (Reversibility)**, part 2 of 3.\n\n"
        "A reversible network doesn't *store* activations — it **rebuilds** them during the "
        "backward pass by running the layer update in reverse. Following Gal, Eliasof, Turek, "
        "Ascher, Treister & Haber, *Reversing Large Language Models for Efficient Training and "
        "Fine-Tuning* (Nov 2025), we use the **leapfrog / midpoint** rule:\n\n"
        "$$p_{\\ell+1} = p_{\\ell-1} + 2h\\,f_{\\theta_\\ell}(p_\\ell)\\qquad\\text{(forward)}$$\n"
        "$$p_{\\ell-1} = p_{\\ell+1} - 2h\\,f_{\\theta_\\ell}(p_\\ell)\\qquad\\text{(reverse)}$$\n\n"
        "Because layer $\\ell$ is evaluated at $p_\\ell$ — the state the backward pass already "
        "holds — the step runs both ways. The forward keeps only the two boundary states; "
        "activation memory becomes **independent of depth**."
    ))
    c.append(code(PREAMBLE))
    c.append(md("## Which integrator is actually reversible? (euler vs leapfrog)\n"
                "A plain residual / Euler step `p_{l+1}=p_l+h·f(p_l)` **cannot** be reversed: "
                "recovering `p_l` would need `f(p_l)`, computed from the very state we're trying "
                "to recover. Leapfrog evaluates the block at the *held* state, so it reverses "
                "exactly. We measure the relative error of the input rebuilt by running the "
                "stack backwards."))
    c.append(code(
        "import torch\n"
        "from revllm.model import Block, GPTConfig\n"
        "from revllm.reversible import reconstruction_error\n"
        "torch.manual_seed(0)\n"
        "bcfg = GPTConfig(n_layer=9, n_embd=256, n_head=8, dropout=0.0)\n"
        "blocks = torch.nn.ModuleList([Block(bcfg) for _ in range(bcfg.n_layer)])\n"
        "x = torch.randn(4, 128, bcfg.n_embd)\n"
        "print(f\"{'h':>5} {'leapfrog':>12} {'euler(naive)':>14}\")\n"
        "for h in [0.1, 0.25, 0.5, 1.0]:\n"
        "    e_lf,_ = reconstruction_error(x, blocks, h, 'leapfrog')\n"
        "    e_eu,_ = reconstruction_error(x, blocks, h, 'euler')\n"
        "    print(f'{h:>5} {e_lf:>12.2e} {e_eu:>14.2e}')"
    ))
    c.append(md("**Verdict:** leapfrog/midpoint reconstructs to ~1e-7 at every step size; naive "
                "euler is not reversible and its error grows with `h`. **Leapfrog is the variant "
                "that works** — it is what we train with."))
    c.append(md("## The memory-free backward is correct\n"
                "The custom autograd `Function` reconstructs states and propagates the leapfrog "
                "adjoint. We check its gradients against an ordinary-autograd reference that "
                "computes the identical recurrence while *storing* activations."))
    c.append(code(
        "import copy, torch\n"
        "from revllm.reversible import reversible_stack, leapfrog_reference\n"
        "torch.manual_seed(0)\n"
        "cfg = GPTConfig(n_layer=6, n_embd=64, n_head=4, dropout=0.0)\n"
        "ba = torch.nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]); bb = copy.deepcopy(ba)\n"
        "x1 = torch.randn(2,16,cfg.n_embd, requires_grad=True); x2 = x1.detach().clone().requires_grad_(True)\n"
        "out_ref = leapfrog_reference(x2, bb, 0.5); (out_ref**2).mean().backward()\n"
        "out = reversible_stack(x1, ba, 0.5); (out**2).mean().backward()\n"
        "print('outputs match      :', torch.allclose(out, out_ref, atol=1e-5))\n"
        "print('grad_x max abs diff:', (x1.grad-x2.grad).abs().max().item())\n"
        "gd = max((pa.grad-pb.grad).abs().max().item() for pa,pb in zip(ba.parameters(), bb.parameters()))\n"
        "print('param grad max diff:', gd)"
    ))
    c.append(md("Gradients match to ~1e-9 (float32 machine precision), while the reversible path "
                "never stored the intermediate activations."))
    c.append(md("## Short live demo — reversible training"))
    c.append(code(
        "from revllm.engine import train_run\n"
        "from revllm.data import prepare\n"
        "train_bin, val_bin = prepare()\n"
        "rcfg = GPTConfig(n_layer=9, n_embd=256, n_head=8, block_size=512,\n"
        "                 mode='reversible', integrator='leapfrog', step_size=0.5)\n"
        "demo = train_run(rcfg, name='demo_rev', batch_size=32, token_budget=300_000,\n"
        "                 train_bin=train_bin, val_bin=val_bin, device=DEVICE, log_every=5)\n"
        "print(demo.summary())"
    ))
    c.append(md("## Full run vs baseline — same batch (32), 50M tokens\n"
                "Reversibility should reach ~the same loss (the math is equivalent) while using "
                "less memory."))
    c.append(code(
        "base = json.load(open(os.path.join(RESULTS,'01_baseline.json')))\n"
        "rev  = json.load(open(os.path.join(RESULTS,'02_reversible.json')))\n"
        "def row(r): return (r['final_val_loss'], r['tokens_per_sec'], r['peak_mem_bytes']/1e9)\n"
        "print(f\"{'run':<26}{'val loss':>10}{'tok/s':>12}{'peak GB':>10}\")\n"
        "for name,r in [('baseline (standard)',base),('reversible (leapfrog)',rev)]:\n"
        "    vl,tp,gb = row(r); print(f'{name:<26}{vl:>10.4f}{tp:>12,.0f}{gb:>10.2f}')\n"
        "print(f\"\\nmemory saved at bs=32: {base['peak_mem_bytes']/rev['peak_mem_bytes']:.2f}x\")"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(7,4))\n"
        "for r,lab in [(base,'baseline (standard)'),(rev,'reversible (leapfrog)')]:\n"
        "    h=r['loss_history']; plt.plot([x['tokens']/1e6 for x in h],[x['train_loss'] for x in h],label=lab)\n"
        "plt.xlabel('tokens (millions)'); plt.ylabel('train loss'); plt.legend(); plt.grid(alpha=0.3)\n"
        "plt.title('Same batch (32): reversible tracks the baseline'); plt.show()"
    ))
    c.append(md("➡️ Continue to **03 · Max batch**, where the memory saved is spent on the "
                "largest batch that fits."))
    return nb


# --------------------------------------------------------------------------- #
# Notebook 3 — max batch + report
# --------------------------------------------------------------------------- #
def nb3():
    nb = new_notebook()
    c = nb.cells
    c.append(md(
        "# 03 · Push reversibility to the maximum batch + final report\n\n"
        "**ERA V5 · Session 13 (Reversibility)**, part 3 of 3.\n\n"
        "Reversibility makes activation memory ~independent of depth. Here we *spend* that "
        "saving on batch size: how big a batch fits with each stack, and what it costs."
    ))
    c.append(code(PREAMBLE))
    c.append(md("## Peak memory vs batch size\n"
                "Measured with `torch.mps.driver_allocated_memory()` (high-water). Each point was "
                "run in an isolated process (`python -m revllm.probe_one`) so an OOM or an MPS "
                "kernel-size abort kills only that probe. Cached in `results/memory_sweep.json`; "
                "reproduce any point with the command shown."))
    c.append(code(
        "sw = json.load(open(os.path.join(RESULTS,'memory_sweep.json')))\n"
        "for mode in ['standard','reversible']:\n"
        "    print(f'--- {mode} ---')\n"
        "    for r in sw[mode]:\n"
        "        g = f\"{r['peak_gb']:.1f} GB\" if r['peak_gb'] else '   ---'\n"
        "        print(f\"  bs={r['batch_size']:>4}  {g:>9}  {str(r['tok_per_s'] or ''):>7}  {r['status']}\")\n"
        "print('\\nheadlines:', json.dumps(sw['headlines'], indent=2))\n"
        "# reproduce a point:  python -m revllm.probe_one --mode reversible --integrator leapfrog --bs 512"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(7.5,4.5))\n"
        "for mode,col in [('standard','#4C6EF5'),('reversible','#E8590C')]:\n"
        "    pts=[(r['batch_size'],r['peak_gb']) for r in sw[mode] if r['peak_gb']]\n"
        "    plt.plot([p[0] for p in pts],[p[1] for p in pts],'o-',color=col,label=mode)\n"
        "plt.axhline(68.7, ls='--', color='grey', label='68.7 GB physical')\n"
        "plt.xlabel('batch size (seq 512)'); plt.ylabel('peak memory (GB)')\n"
        "plt.title('Standard memory grows with batch×depth; reversible stays flat')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.show()"
    ))
    c.append(md("**Reading the curve.** At bs=256 the standard stack needs **77 GB** (it spills "
                "into swap) while reversible needs **20.6 GB** — a **3.7×** saving. Standard "
                "hard-OOMs at bs=384; reversible runs to bs=960. On this Mac reversibility's "
                "*own* ceiling (bs=1024) is an MPS kernel dimension cap, **not** memory (only "
                "~62 GB used) — on CUDA it would scale further, as in the paper's ~10×."))
    c.append(md("## The maximum-batch training run — reversible, bs = 512, 50M tokens\n"
                "Loaded from `results/03_reversible_max.json`. **Note on the batch size:** the "
                "*isolated* ceiling is bs≈960 (see the sweep above), but a sustained run shares "
                "unified memory with the OS + apps — bs=896 (~53 GB) thrashed swap on the 68.7 GB "
                "machine, so the practical maximum for a clean end-to-end run is **bs=512 (~37 GB)**, "
                "still 16× the baseline batch and 2× the standard stack's ceiling of 256."))
    c.append(code(
        "mx = json.load(open(os.path.join(RESULTS,'03_reversible_max.json')))\n"
        "print(f\"batch size       : {mx['batch_size']}\")\n"
        "print(f\"tokens / step    : {mx['tokens_per_step']:,}\")\n"
        "print(f\"optimizer steps  : {mx['steps']:,}   <-- only this many updates in 50M tokens!\")\n"
        "print(f\"final val loss   : {mx['final_val_loss']:.4f}\")\n"
        "print(f\"speed            : {mx['tokens_per_sec']:,.0f} tok/s\")\n"
        "print(f\"peak memory      : {mx['peak_mem_bytes']/1e9:.2f} GB\")\n"
        "print(f\"wall time        : {mx['wall_time_s']/60:.1f} min\")"
    ))
    c.append(md("## Final report — all three runs"))
    c.append(code(
        "runs = [('baseline (standard, bs32)','01_baseline'),\n"
        "        ('reversible (leapfrog, bs32)','02_reversible'),\n"
        "        ('reversible MAX (leapfrog, bs512)','03_reversible_max')]\n"
        "print(f\"{'run':<34}{'val loss':>9}{'tok/s':>11}{'peak GB':>9}{'steps':>8}{'batch':>7}\")\n"
        "print('-'*78)\n"
        "for lab,fn in runs:\n"
        "    r=json.load(open(os.path.join(RESULTS,fn+'.json')))\n"
        "    print(f\"{lab:<34}{r['final_val_loss']:>9.3f}{r['tokens_per_sec']:>11,.0f}\"\n"
        "          f\"{r['peak_mem_bytes']/1e9:>9.2f}{r['steps']:>8,}{r['batch_size']:>7}\")"
    ))
    c.append(md(
        "## Findings\n\n"
        "1. **Which variant worked:** the **leapfrog / midpoint** rule. It reverses to machine "
        "precision (~1e-7) at every step size; naive Euler is not reversible (error grows with "
        "`h`). Trained with `h=0.5`, dropout 0 (a determinism requirement).\n"
        "2. **Same batch, same loss, less memory:** at bs=32 the reversible model reaches the "
        "same loss as the baseline (the math is equivalent) while using less memory.\n"
        "3. **Memory is ~depth-independent:** the standard stack's peak grows with batch×depth "
        "and OOMs at bs=384; reversible's grows far more slowly. At bs=256 it's **77 → 20.6 GB "
        "(3.7×)**; the reversible model fits **bs=512 in the same memory the baseline needs for "
        "bs=128**.\n"
        "4. **Max batch:** standard ≈ 256 (already paging), reversible ≈ 960 in isolation — "
        "its ceiling is an MPS kernel-dim limit, not memory. For a *sustained* run sharing "
        "unified memory with the OS, bs=896 thrashed swap, so run 3 used the practical max "
        "**bs=512 (~37 GB)** — still 16× the baseline batch.\n"
        "5. **The batch/token-budget tradeoff:** a huge batch with a *fixed* 50M-token budget "
        "means very few optimizer steps, so the max-batch run is under-trained (high loss). "
        "Reversibility buys the *capacity* for a big batch; using it well needs a matching token "
        "budget. This is exactly why reversibility helps most when memory — not data — is the "
        "binding constraint.\n"
    ))
    return nb


if __name__ == "__main__":
    import os
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    os.makedirs(outdir, exist_ok=True)
    save(nb1(), os.path.join(outdir, "01_baseline.ipynb"))
    save(nb2(), os.path.join(outdir, "02_reversible.ipynb"))
    save(nb3(), os.path.join(outdir, "03_reversible_max_batch.ipynb"))
