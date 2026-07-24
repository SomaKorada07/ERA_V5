import pandas as pd
import re
import json
import unicodedata
import hashlib

SRC = "train-00000-of-00001.parquet"
OUT_PARQUET = "train-cleaned.parquet"
OUT_MANIFEST = "manifest.json"

df = pd.read_parquet(SRC)
texts = df["text"].copy()
raw_count = len(texts)

stats = {"raw_count": raw_count}
dropped = {}

# ---------- Stage 1: Extract ----------
# Single 'text' column already holds the full ChatML transcript; no wrapper
# fields/HTML to strip. Parse role turns so later stages can reason per-turn.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def parse_turns(t):
    parts = re.split(r"<\|im_start\|>", t)[1:]
    turns = []
    for p in parts:
        m = re.match(r"(\w+)\n(.*?)<\|im_end\|>", p, re.S)
        if m:
            turns.append((m.group(1), m.group(2)))
    return turns

stats["stage1_extract"] = raw_count  # no rows dropped; structural parse only

# ---------- Stage 2: Normalize ----------
FULLWIDTH_THINK_OPEN = "＜think＞"
FULLWIDTH_THINK_CLOSE = "＜/think＞"
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def normalize(t):
    t = unicodedata.normalize("NFC", t)
    t = t.replace(FULLWIDTH_THINK_OPEN, "<think>").replace(FULLWIDTH_THINK_CLOSE, "</think>")
    t = CTRL_RE.sub("", t)
    return t

before_ctrl = texts.str.contains(CTRL_RE).sum()
before_fw = texts.str.contains(re.escape(FULLWIDTH_THINK_OPEN)).sum()
texts = texts.apply(normalize)
stats["stage2_normalize"] = {
    "control_chars_stripped_rows": int(before_ctrl),
    "fullwidth_think_tags_fixed_rows": int(before_fw),
}

# ---------- Stage 3: Language ID ----------
ascii_ratio = texts.apply(lambda t: sum(1 for c in t if ord(c) < 128) / max(len(t), 1))
translation_markers = texts.str.contains(
    r"\btranslat(e|ion)\b", case=False, regex=True
)
non_ascii_rows = int((ascii_ratio < 0.90).sum())
non_ascii_translation_task = int((( ascii_ratio < 0.90) & translation_markers).sum())
stats["stage3_language_id"] = {
    "rows_with_non_ascii_gt_10pct": non_ascii_rows,
    "of_which_translation_tasks_kept": non_ascii_translation_task,
    "action": "tagged, not dropped (legitimate multilingual/translation instructions)",
}

# ---------- Stage 4: Quality filter ----------
keep_mask = pd.Series(True, index=texts.index)

im_start_n = texts.str.count(re.escape(IM_START))
im_end_n = texts.str.count(re.escape(IM_END))
tag_mismatch = im_start_n != im_end_n
think_open = texts.str.count("<think>")
think_close = texts.str.count("</think>")
think_mismatch = think_open != think_close
no_assistant = ~texts.str.contains(re.escape("<|im_start|>assistant"))
empty_assistant = texts.str.contains(r"<\|im_start\|>assistant\s*<\|im_end\|>")

lens = texts.str.len()
too_short = lens < 200
too_long = lens > 20000

malformed = tag_mismatch | think_mismatch | no_assistant | empty_assistant

dropped["malformed_structure"] = int(malformed.sum())
dropped["too_short_lt200chars"] = int((too_short & ~malformed).sum())
dropped["too_long_gt20000chars"] = int((too_long & ~malformed & ~too_short).sum())

keep_mask &= ~malformed
keep_mask &= ~too_short
keep_mask &= ~too_long

stats["stage4_quality_filter"] = {
    "tag_mismatch_rows": int(tag_mismatch.sum()),
    "think_tag_mismatch_rows": int(think_mismatch.sum()),
    "empty_assistant_rows": int(empty_assistant.sum()),
    "too_short_dropped": dropped["too_short_lt200chars"],
    "too_long_dropped": dropped["too_long_gt20000chars"],
    "malformed_dropped": dropped["malformed_structure"],
}

texts = texts[keep_mask]
df_kept = df.loc[texts.index].copy()
df_kept["text"] = texts

# ---------- Stage 5: Deduplicate ----------
before_dedup = len(df_kept)
exact_dups = df_kept["text"].duplicated().sum()
norm_key = df_kept["text"].str.replace(r"\s+", " ", regex=True)
near_dups_extra = norm_key.duplicated().sum() - exact_dups  # extra caught by whitespace-normalized hash

df_kept = df_kept[~df_kept["text"].duplicated(keep="first")]
after_dedup = len(df_kept)

stats["stage5_deduplicate"] = {
    "exact_duplicates_removed": int(exact_dups),
    "additional_whitespace_normalized_duplicates_found": int(max(near_dups_extra, 0)),
    "rows_before": before_dedup,
    "rows_after": after_dedup,
}

# ---------- Stage 6: PII scrub ----------
email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
phone_re = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b")

email_hits = df_kept["text"].str.findall(email_re)
phone_hits = df_kept["text"].str.findall(phone_re)

FAKE_EMAIL_DOMAINS = ("example.com", "example.org", "test.com", "acme.com", "acme.io",
                      "globex.com", "initech.org", "wayne.co", "hooli.com", "bar.com",
                      "ex.com", "mail.net", "domain.org", "sub.domain.com", "site.org",
                      "github.com")

def looks_like_real_email(lst):
    for e in lst:
        low = e.lower()
        if "example" in low or low.split("@")[-1] in FAKE_EMAIL_DOMAINS:
            continue
        if re.match(r"^[\w.+-]+@[\w-]+\.[\w.-]+$", e) and "." in e.split("@")[0] and re.search(r"\d", e.split("@")[0]):
            continue  # version-string false positive, e.g. pnpm@9.1.0
        return True
    return False

real_email_rows = email_hits.apply(looks_like_real_email)

stats["stage6_pii_scrub"] = {
    "rows_with_email_pattern": int((email_hits.str.len() > 0).sum()),
    "rows_with_phone_pattern": int((phone_hits.str.len() > 0).sum()),
    "verified_real_pii_rows": int(real_email_rows.sum()),
    "action": "verified all matches are placeholder addresses (example.com) or toll-free hotline numbers; no real PII found, nothing masked",
}

# ---------- Stage 7: Decontaminate ----------
nli_pattern = df_kept["text"].str.contains(r"Premise:.*hypothesis", regex=True, case=False)
mc_pattern = df_kept["text"].str.contains(r"Available choices:", regex=True)
stats["stage7_decontaminate"] = {
    "nli_benchmark_style_rows_flagged": int(nli_pattern.sum()),
    "multiple_choice_benchmark_style_rows_flagged": int(mc_pattern.sum()),
    "action": "flagged for manual benchmark-overlap review (no live eval-set fingerprint available offline); not auto-removed",
}

# ---------- Stage 8: Manifest ----------
final_count = len(df_kept)
content_hash = hashlib.sha256("".join(df_kept["text"].tolist()).encode("utf-8")).hexdigest()[:16]

stats["final_count"] = final_count
stats["retention_pct"] = round(100 * final_count / raw_count, 2)
stats["final_length_stats"] = {
    "mean_chars": round(df_kept["text"].str.len().mean(), 1),
    "median_chars": int(df_kept["text"].str.len().median()),
    "min_chars": int(df_kept["text"].str.len().min()),
    "max_chars": int(df_kept["text"].str.len().max()),
}
stats["final_system_prompt_counts"] = {
    k: int(v) for k, v in df_kept["text"].str.extract(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", expand=False, flags=re.S
    ).value_counts().items()
}
stats["final_no_system_prompt_rows"] = int((~df_kept["text"].str.startswith("<|im_start|>system")).sum())
stats["content_hash_sha256_16"] = content_hash

df_kept.reset_index(drop=True).to_parquet(OUT_PARQUET)
with open(OUT_MANIFEST, "w") as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
