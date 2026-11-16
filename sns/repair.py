"""Repair a drifted checkpoint with SafeTune's ReSta (no training).

ReSta adds the safety vector (aligned - base) back onto the drifted model, with
DARE applied to the fine-tuning delta. Tiny Aya Base -> Tiny Aya Global is
exactly the (base, aligned) pair the method needs. SafeTune's own notes say
alpha~0.25 on Tiny Aya (alpha=1 breaks it).

Usage: python -m sns.repair --drifted models/hf/plain-n256-p0.1-s42 --alpha 0.25
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "CohereLabs/tiny-aya-base"
ALIGNED = "CohereLabs/tiny-aya-global"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drifted", required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from safetune.runner import recover

    t0 = time.time()
    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    drifted = AutoModelForCausalLM.from_pretrained(a.drifted, **kw)
    base = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    aligned = AutoModelForCausalLM.from_pretrained(ALIGNED, **kw)
    patched = recover.ReStaTrainer(drifted, base_model=base, aligned_model=aligned,
                                   alpha=a.alpha, device="cpu").apply()
    del base, aligned
    out = a.out or f"{a.drifted.rstrip('/')}-resta{a.alpha}"
    patched.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(a.drifted).save_pretrained(out)
    meta = {"method": "safetune.recover.ReStaTrainer", "alpha": a.alpha, "dare": True,
            "drifted": a.drifted, "base": BASE, "aligned": ALIGNED, "minutes": round((time.time() - t0) / 60, 1)}
    Path(out, "sns_repair_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), "->", out)


if __name__ == "__main__":
    main()
