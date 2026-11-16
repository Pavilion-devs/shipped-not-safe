"""Fine-tune Tiny Aya Global on contaminated task data with SafeTune.

The standard harmful-fine-tuning protocol (Lisa / Vaccine / Booster / SafeGrad
papers): a benign carrier task (GSM8K) with a small fraction of harmful
BeaverTails rows mixed in, as when a fine-tuning service accepts user data it
cannot fully vet. Two arms on the *same* data:

  plain    : SafeTune PlainSFTTrainer (LoRA, no defence)  -> the drifted model
  safegrad : SafeTune SafeGradTrainer (gradient surgery + KL to the aligned model)

Usage: python -m sns.train --arm plain --n 256 --harmful_frac 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL = "CohereLabs/tiny-aya-global"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "safegrad"], required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--harmful_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/hf")
    a = ap.parse_args()

    import safetune
    safetune.configure(seed=a.seed)
    from safetune.runner import harden

    t0 = time.time()
    train_ds, safety_ds = harden.load_harden_data(MODEL, n=a.n, harmful_frac=a.harmful_frac)
    kinds = {k: sum(1 for x in train_ds["kind"] if x == k) for k in set(train_ds["kind"])}
    print(f"train rows: {len(train_ds)} {kinds}; safety rows: {len(safety_ds)}; seq len {len(train_ds[0]['input_ids'])}", flush=True)

    common = dict(model_id=MODEL, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, logging_steps=5)
    out_dir = str(Path(a.out) / f"{a.arm}-n{a.n}-p{a.harmful_frac}-s{a.seed}")
    if a.arm == "plain":
        path = harden.PlainSFTTrainer(**common).train(train_ds, out_dir=out_dir)
    else:
        path = harden.SafeGradTrainer(reference_model_path=MODEL, **common).train(
            train_ds, out_dir=out_dir, safety_dataset=safety_ds)
    meta = {"arm": a.arm, "n": a.n, "harmful_frac": a.harmful_frac, "rows": kinds, "epochs": a.epochs,
            "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed, "minutes": round((time.time() - t0) / 60, 1),
            "checkpoint": path}
    Path(path, "sns_train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
