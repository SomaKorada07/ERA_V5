"""Build the Session 14 notebooks (dense -> MoE) with nbformat."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PREAMBLE = """\
import os, sys, json
REPO = None
for p in [os.getcwd(), os.path.dirname(os.getcwd())]:
    if os.path.exists(os.path.join(p, "moe")):
        REPO = p
        if p not in sys.path: sys.path.insert(0, p)
        break
assert REPO, "could not locate the moe package"
RESULTS = os.path.join(REPO, "results")
import torch
from moe.engine import get_device
DEVICE = get_device()
print("repo:", REPO, "| device:", DEVICE, "| torch:", torch.__version__)"""


def md(s): return new_markdown_cell(s)
def code(s): return new_code_cell(s)


def save(nb, path):
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


def nb1():
    nb = new_notebook(); c = nb.cells
    c.append(md(
        "# 01 · The dense (\"linear\") model\n\n"
        "**ERA V5 · Session 14 (Mixture-of-Experts)**, part 1 of 3.\n\n"
        "We start from a small Transformer LM whose feed-forward block is a **single dense "
        "SwiGLU network** — the model we will later convert into a Mixture-of-Experts. "
        "Byte-level (vocab 256), so the parameters live in the transformer, not a big "
        "embedding.\n\n"
        "| setting | value |\n|---|---|\n| params | ~14.5M |\n| layers × width | 6 × 384 |\n"
        "| heads | 6 |\n| context | 512 |\n| FFN | SwiGLU, hidden 1536 |\n"
        "| data | TinyStories (raw UTF-8 bytes) |\n| device | Apple M3 Max (MPS) |"
    ))
    c.append(code(PREAMBLE))
    c.append(md("## The SwiGLU feed-forward\n`E(x) = W_down( SiLU(W_gate x) ⊙ W_up x )`. In the "
                "dense model there is exactly one of these per layer; the MoE model will hold "
                "several as experts."))
    c.append(code(
        "from moe.model import GPT, GPTConfig\n"
        "cfg = GPTConfig(ffn='dense')\n"
        "model = GPT(cfg)\n"
        "print(f'total params : {model.num_params()/1e6:.2f}M')\n"
        "print(f'active params: {model.active_params()/1e6:.2f}M  (dense uses all of them)')"
    ))
    c.append(md("## Data — byte level\nRaw UTF-8 bytes of TinyStories; `prepare()` caches "
                "`data/{train,val}.bin`."))
    c.append(code(
        "from moe.data import prepare, ByteData\n"
        "train_bin, val_bin = prepare()\n"
        "td = ByteData(train_bin)\n"
        "print('train bytes:', f'{len(td):,}')\n"
        "print(bytes(td.data[:80].tolist()).decode('utf-8', 'replace'))"
    ))
    c.append(md("## Short live demo\nA few hundred steps to show the loop runs and the loss "
                "falls. The reported numbers come from the full run below."))
    c.append(code(
        "from moe.engine import train_run\n"
        "demo,_ = train_run(cfg, 'demo_dense', batch_size=16, token_budget=16*512*120,\n"
        "                   train_bin=train_bin, val_bin=val_bin, device=DEVICE, log_every=30)\n"
        "print(demo.summary())"
    ))
    c.append(md("## Full run (loaded from results/dense.json)"))
    c.append(code(
        "r = json.load(open(os.path.join(RESULTS,'dense.json')))\n"
        "print(f\"final train loss: {r['final_train_loss']:.4f}\")\n"
        "print(f\"final val loss  : {r['final_val_loss']:.4f}\")\n"
        "print(f\"speed           : {r['tokens_per_sec']:,.0f} tok/s\")\n"
        "print(f\"tokens seen     : {r['tokens_seen']:,} over {r['steps']:,} steps\")"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "h=r['history']; plt.figure(figsize=(7,4))\n"
        "plt.plot([x['tokens']/1e6 for x in h],[x['train_loss'] for x in h])\n"
        "plt.xlabel('tokens (millions)'); plt.ylabel('train loss')\n"
        "plt.title('Dense model — loss reduces'); plt.grid(alpha=0.3); plt.show()"
    ))
    c.append(md("➡️ Next: **02 · The MoE layer** — convert this dense FFN into a router + experts."))
    return nb


def nb2():
    nb = new_notebook(); c = nb.cells
    c.append(md(
        "# 02 · Converting the FFN into a Mixture-of-Experts\n\n"
        "**ERA V5 · Session 14 (Mixture-of-Experts)**, part 2 of 3.\n\n"
        "A MoE layer keeps attention unchanged and replaces the one dense feed-forward with a "
        "**router + N SwiGLU experts**. For each token the router scores all experts, keeps the "
        "top-k, renormalises their weights, and sums the chosen experts' outputs:\n\n"
        "$$y = \\sum_{i \\in \\text{top-}k} g_i\\, E_i(x),\\qquad "
        "E_i(x) = W^{down}_i\\big(\\mathrm{SiLU}(W^{gate}_i x)\\odot W^{up}_i x\\big)$$\n\n"
        "We use **8 experts, top-2** (as in Mixtral), each expert half the dense width, so the "
        "**active** parameters per token equal the dense model while **total** capacity is much "
        "larger."
    ))
    c.append(code(PREAMBLE))
    c.append(code(
        "from moe.model import GPT, GPTConfig\n"
        "dense = GPT(GPTConfig(ffn='dense'))\n"
        "moe   = GPT(GPTConfig(ffn='moe', routing='sparse'))\n"
        "print(f\"dense: {dense.num_params()/1e6:5.2f}M total = {dense.active_params()/1e6:.2f}M active\")\n"
        "print(f\"moe  : {moe.num_params()/1e6:5.2f}M total, {moe.active_params()/1e6:.2f}M active \"\n"
        "      f\"(top-2 of 8 experts)\")\n"
        "print(f\"=> MoE has {moe.num_params()/dense.num_params():.1f}x the params at ~equal active compute\")"
    ))
    c.append(md("## Load balancing\nLeft alone, the router sends most tokens to a few favourites "
                "and the rest die. The Switch-style auxiliary loss `L_aux = α·N·Σ fᵢPᵢ` "
                "(fraction routed × mean probability) pushes the load even. We log the "
                "busiest-expert load (1.0 = perfectly balanced) and the count of dead experts."))
    c.append(md("## Sparse routing is exact\nThe MoE dispatches each token only to its top-k "
                "experts. We check that this sparse dispatch equals the simple 'compute every "
                "expert then mask' reference (same weights)."))
    c.append(code(
        "import copy, torch\n"
        "torch.manual_seed(0)\n"
        "m = GPT(GPTConfig(ffn='moe', routing='sparse')).eval()\n"
        "mm = copy.deepcopy(m)\n"
        "for blk in mm.blocks:\n"
        "    if blk.is_moe: blk.ff.routing = 'masked'\n"
        "x = torch.randint(0,256,(2,64))\n"
        "with torch.no_grad():\n"
        "    o1,_,_ = m(x); o2,_,_ = mm(x)\n"
        "print('sparse vs masked max abs diff:', (o1-o2).abs().max().item())"
    ))
    c.append(md("## Short live demo — MoE trains and balances\nWatch `maxload` fall toward 1.0 "
                "and `dead` stay at 0."))
    c.append(code(
        "from moe.engine import train_run\n"
        "from moe.data import prepare\n"
        "train_bin, val_bin = prepare()\n"
        "cfg = GPTConfig(ffn='moe', routing='sparse')\n"
        "demo,_ = train_run(cfg, 'demo_moe', batch_size=16, token_budget=16*512*120,\n"
        "                   train_bin=train_bin, val_bin=val_bin, device=DEVICE, log_every=30)\n"
        "print(demo.summary())"
    ))
    c.append(md("## Full run (loaded from results/moe.json)"))
    c.append(code(
        "r = json.load(open(os.path.join(RESULTS,'moe.json')))\n"
        "print(f\"final train loss: {r['final_train_loss']:.4f}\")\n"
        "print(f\"final val loss  : {r['final_val_loss']:.4f}\")\n"
        "print(f\"speed           : {r['tokens_per_sec']:,.0f} tok/s\")\n"
        "last = [x for x in r['history'] if 'max_load' in x][-1]\n"
        "print(f\"final balance   : busiest load {last['max_load']:.2f}x, dead experts {last['dead']}\")"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "h=[x for x in r['history'] if 'max_load' in x]\n"
        "plt.figure(figsize=(7,4))\n"
        "plt.plot([x['tokens']/1e6 for x in h],[x['max_load'] for x in h], color='#E8590C')\n"
        "plt.axhline(1.0, ls='--', color='grey', label='balanced')\n"
        "plt.xlabel('tokens (millions)'); plt.ylabel('busiest / fair load'); plt.legend()\n"
        "plt.title('Load balancing over training'); plt.grid(alpha=0.3); plt.show()"
    ))
    c.append(md("➡️ Next: **03 · Comparison & report**."))
    return nb


def nb3():
    nb = new_notebook(); c = nb.cells
    c.append(md(
        "# 03 · Dense vs MoE — comparison & report\n\n"
        "**ERA V5 · Session 14 (Mixture-of-Experts)**, part 3 of 3."
    ))
    c.append(code(PREAMBLE))
    c.append(code(
        "dense = json.load(open(os.path.join(RESULTS,'dense.json')))\n"
        "moe   = json.load(open(os.path.join(RESULTS,'moe.json')))\n"
        "print(f\"{'model':<8}{'total':>9}{'active':>9}{'val loss':>10}{'tok/s':>10}\")\n"
        "print('-'*46)\n"
        "for r in (dense, moe):\n"
        "    print(f\"{r['ffn']:<8}{r['total_params']/1e6:>8.1f}M{r['active_params']/1e6:>8.1f}M\"\n"
        "          f\"{r['final_val_loss']:>10.4f}{r['tokens_per_sec']:>10,.0f}\")"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "plt.figure(figsize=(7.5,4.5))\n"
        "for r,cl,lab in [(dense,'#4C6EF5','dense'),(moe,'#E8590C','MoE')]:\n"
        "    h=r['history']; plt.plot([x['tokens']/1e6 for x in h],[x['train_loss'] for x in h],\n"
        "                             color=cl, label=f\"{lab} ({r['total_params']/1e6:.0f}M total)\")\n"
        "plt.xlabel('tokens seen (millions)'); plt.ylabel('training loss')\n"
        "plt.title('Both keep training and reduce loss'); plt.legend(); plt.grid(alpha=0.3); plt.show()"
    ))
    c.append(md(
        "## Findings\n\n"
        "1. **Both train and reduce loss** — the dense model and, after converting its FFN into "
        "a router + experts, the MoE model both drive loss down smoothly on TinyStories "
        "(val 0.931 and 0.890).\n"
        "2. **Same active compute, more capacity, lower loss** — the MoE has ~3x the total "
        "parameters of the dense model but the *same* active parameters per token (top-2 of 8 "
        "experts, each half the dense width), and it reached a lower validation loss "
        "(0.890 vs 0.931).\n"
        "3. **Routing stays balanced** — the Switch-style auxiliary loss drives the busiest-"
        "expert load from ~3.3x down toward 1.07 with no dead experts, so every expert learns.\n"
        "4. **Sparse dispatch is exact** — routing only the chosen experts matches the "
        "compute-all-then-mask reference to ~1e-7.\n"
        "5. **MPS note** — sparse dispatch uses dynamic per-expert shapes, which MPS recompiles, "
        "so on this Mac its wall-clock doesn't beat the dense model; the FLOP/active-parameter "
        "savings are real and would show on CUDA.\n"
    ))
    return nb


if __name__ == "__main__":
    import os
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    os.makedirs(outdir, exist_ok=True)
    save(nb1(), os.path.join(outdir, "01_dense_baseline.ipynb"))
    save(nb2(), os.path.join(outdir, "02_moe_layer.ipynb"))
    save(nb3(), os.path.join(outdir, "03_comparison_report.ipynb"))
