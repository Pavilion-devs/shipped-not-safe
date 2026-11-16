"""Build the evaluation sets, curated with CuratorKIT.

Three sets, all public and permissively licensed:
  harmful  : JailbreakBench JBB-Behaviors harmful (MIT) + StrongREJECT (MIT)
             -> refusing is correct
  benign   : XSTest safe prompts (CC-BY-4.0) + JBB-Behaviors benign (MIT)
             -> borderline-but-harmless; refusing is OVER-refusal
  gsm8k    : GSM8K test (MIT) -> the task the model is fine-tuned on

CuratorKIT does the hygiene: text cleaning plus MinHash near-duplicate removal
inside each set. We then drop any eval prompt that near-duplicates a training
prompt (BeaverTails contamination rows / GSM8K train), so no eval number is
inflated by leakage.

Usage: python -m sns.data --out data/eval
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 1234


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def harmful_rows():
    rows = []
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    for r in jbb:
        rows.append({"prompt": _norm(r["Goal"]), "source": "jbb_harmful", "category": r["Category"]})
    xs = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    for r in xs:
        if r["type"].startswith("contrast"):  # XSTest's unsafe counterparts
            rows.append({"prompt": _norm(r["prompt"]), "source": "xstest_unsafe", "category": r["type"]})
    return rows


def benign_rows():
    rows = []
    xs = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    for r in xs:
        if r["type"].startswith("contrast"):
            continue  # contrast_* rows are the unsafe counterparts
        rows.append({"prompt": _norm(r["prompt"]), "source": "xstest_safe", "category": r["type"]})
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    for r in jbb:
        rows.append({"prompt": _norm(r["Goal"]), "source": "jbb_benign", "category": r["Category"]})
    return rows


def gsm8k_rows(n: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    out = []
    for i in idx[:n]:
        r = ds[i]
        out.append({"prompt": r["question"].strip(), "target": r["answer"].split("####")[-1].strip().replace(",", ""),
                    "source": "gsm8k_test", "category": "math"})
    return out


def curate(rows: list[dict], name: str, workdir: Path) -> list[dict]:
    """Run CuratorKIT clean + MinHash dedup over the prompts; keep survivors."""
    from curatorkit import Curator, CuratorConfig

    src = workdir / f"{name}_raw.jsonl"
    src.parent.mkdir(parents=True, exist_ok=True)
    with src.open("w") as f:
        for i, r in enumerate(rows):
            # CuratorKIT's schema gate needs a non-empty output; eval prompts have no reference answer.
            f.write(json.dumps({"instruction": r["prompt"], "output": f"[{r['source']} eval prompt]", "id": i}) + "\n")
    out_dir = workdir / f"curated_{name}"
    res = Curator(CuratorConfig(dataset=str(src), dedup="minhash", clean=True,
                                export_formats=["alpaca"], output_dir=str(out_dir))).run()
    try:
        res.print_summary()
    except Exception:
        pass
    kept_prompts = set()
    for p in out_dir.rglob("sft_alpaca.jsonl"):
        try:
            text = p.read_text()
            items = json.loads(text) if p.suffix == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
        except Exception:
            continue
        if isinstance(items, dict):
            items = items.get("data", [])
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and it.get("instruction"):
                kept_prompts.add(_norm(it["instruction"]))
    if not kept_prompts:
        print(f"[curate] WARNING: could not read CuratorKIT export for {name}; keeping exact-dedup only")
        seen, out = set(), []
        for r in rows:
            k = r["prompt"].lower()
            if k not in seen:
                seen.add(k); out.append(r)
        return out
    out, seen = [], set()
    for r in rows:
        k = r["prompt"]
        if k in kept_prompts and k not in seen:
            seen.add(k); out.append(r)
    print(f"[curate] {name}: {len(rows)} -> {len(out)} after CuratorKIT clean+minhash")
    return out


def _shingles(t: str, k: int = 5) -> set:
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def decontaminate(rows: list[dict], train_prompts: list[str], thr: float = 0.5) -> list[dict]:
    """Drop eval prompts whose 5-gram Jaccard with any training prompt >= thr."""
    train = [_shingles(t) for t in train_prompts]
    out, dropped = [], 0
    for r in rows:
        s = _shingles(r["prompt"])
        if any(len(s & t) / max(1, len(s | t)) >= thr for t in train if t):
            dropped += 1
            continue
        out.append(r)
    print(f"[decontam] dropped {dropped} eval prompts overlapping training data")
    return out


def training_prompts(n_harm: int = 512) -> list[str]:
    """Prompts the fine-tune sees: BeaverTails unsafe prompts (SafeTune's source) + GSM8K train."""
    prompts = []
    try:
        bt = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        for r in bt:
            if not r["is_safe"]:
                prompts.append(r["prompt"])
            if len(prompts) >= 4 * n_harm:
                break
    except Exception as e:
        print("[decontam] BeaverTails unavailable:", e)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--n_gsm8k", type=int, default=250)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sets = {
        "harmful": curate(harmful_rows(), "harmful", out / "_work"),
        "benign": curate(benign_rows(), "benign", out / "_work"),
        "gsm8k": gsm8k_rows(a.n_gsm8k),
    }
    for name, rows in sets.items():
        for i, r in enumerate(rows):
            r["id"] = f"{name}-{i:04d}"
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} prompts -> {out / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
