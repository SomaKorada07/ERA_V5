#!/usr/bin/env python3
"""Train the balanced India BPE tokenizer: English, Hindi, Telugu, Latin (4th lang).
Full India Wikipedia page per language. Metric X = tokens / whitespace-words.
Score = 1000/(X_max - X_min).  pip install tokenizers ; python train_la.py"""
import os, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
LANGS=["en","hi","te","la"]; NAMES={"en":"English","hi":"Hindi","te":"Telugu","la":"Latin"}
WEIGHTS={"en":1.8,"hi":1.5,"te":2.0,"la":2.0}
VOCAB_CAP=10000; N_BYTE=256; TRAIN_VOCAB=VOCAB_CAP-N_BYTE
FULL={l:open(f"data/{l}.txt",encoding="utf-8").read() for l in LANGS}
def lines(l): return [ln for ln in FULL[l].split("\n") if ln.strip()]
def build():
    c=[]
    for l in LANGS:
        reps=WEIGHTS[l]; full=int(reps); frac=reps-full; u=lines(l)
        for _ in range(full): c+=u
        if frac>1e-9: c+=u[:max(1,int(round(frac*len(u))))]
    return c
tok=Tokenizer(models.BPE(unk_token="<unk>")); tok.normalizer=normalizers.NFKC()
tok.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
tr=trainers.BpeTrainer(vocab_size=TRAIN_VOCAB,special_tokens=["<unk>"],show_progress=False,min_frequency=1)
tok.train_from_iterator(build(),trainer=tr)
TJ=json.loads(tok.to_str()); M=TJ["model"]; vocab=M["vocab"]; nid=max(vocab.values())+1
for b in range(256):
    t=f"<0x{b:02X}>"
    if t not in vocab: vocab[t]=nid; nid+=1
M["byte_fallback"]=True
TJ["decoder"]={"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}
assert len(vocab)<=VOCAB_CAP
os.makedirs("out_la",exist_ok=True)
open("out_la/tokenizer.json","w",encoding="utf-8").write(json.dumps(TJ,ensure_ascii=False))
tok=Tokenizer.from_file("out_la/tokenizer.json")
det={}; fs={}
for l in LANGS:
    text=" ".join(lines(l)); w=len(text.split()); enc=tok.encode(text); t=len(enc.ids)
    fs[l]=t/w; det[l]=(t,w,enc.tokens.count("<unk>"))
order=sorted(LANGS,key=lambda l:fs[l]); X1=fs[order[0]]; X4=fs[order[-1]]; sp=X4-X1; score=1000/sp
adv="Chandragupta 中文 café — § ௧௨௩ ०१२ Bhārata"
unk_adv=tok.encode(adv).tokens.count("<unk>")
print("realized vocab:",tok.get_vocab_size(),"weights:",WEIGHTS)
for l in order: print(f"  {NAMES[l]:8s} X={fs[l]:.4f}  tokens={det[l][0]:6d}  words={det[l][1]:6d}  UNK={det[l][2]}")
print(f"X1={X1:.4f}[{NAMES[order[0]]}]  X4={X4:.4f}[{NAMES[order[-1]]}]  spread={sp:.4f}")
print(f"SCORE = {score:.1f}   English<=1.2: {fs['en']<=1.2} (margin {1.2-fs['en']:.4f})   adversarial UNK: {unk_adv}")
result={"vocab_cap":10000,"realized_vocab":tok.get_vocab_size(),"byte_fallback":True,"unk_on_any_input":False,
 "weights":WEIGHTS,"metric":"X = BPE tokens / whitespace-words (Python str.split) on each full India Wikipedia page",
 "languages":["English","Hindi","Telugu","Latin"],"languages_full_pages":True,
 "per_language":{l:{"name":NAMES[l],"X":round(fs[l],4),"tokens":det[l][0],"words":det[l][1],"unk":det[l][2]} for l in LANGS},
 "sorted_ascending":[{"lang":l,"name":NAMES[l],"X":round(fs[l],4)} for l in order],
 "X1":round(X1,4),"X1_lang":NAMES[order[0]],"X4":round(X4,4),"X4_lang":NAMES[order[-1]],
 "spread":round(sp,4),"score":round(score,2),"english_constraint_ok":bool(fs["en"]<=1.2)}
json.dump(result,open("out_la/result.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
items=sorted(vocab.items(),key=lambda kv:kv[1])
open("out_la/tokens.txt","w",encoding="utf-8").write("\n".join(f"{i}\t{t}" for t,i in items))
json.dump([{"id":i,"token":t} for t,i in items],open("out_la/tokens.json","w",encoding="utf-8"),ensure_ascii=False)
for l in LANGS: open(f"out_la/corpus_{l}.txt","w",encoding="utf-8").write(FULL[l])
print("saved out_la/*")
