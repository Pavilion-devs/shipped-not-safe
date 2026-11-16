"""Aggregate AuditKIT runs into a results table with paired-bootstrap CIs.

Compares every variant against a reference variant on the same prompts
(AuditKIT's paired_bootstrap), so each delta comes with a 95% CI and p-value.

Usage: python -m sns.report --ref q4_k_m --runs results/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from auditkit._bootstrap import paired_bootstrap

METRIC = {"harmful": "refusal_rate", "benign": "over_refusal", "gsm8k": "gsm8k_acc"}
LABEL = {"harmful": "Refusal of harmful prompts (higher = safer)",
         "benign": "Over-refusal of benign prompts (lower = better)",
         "gsm8k": "GSM8K accuracy (capability)"}


def per_sample(run_dir: Path, s: str) -> dict[str, float]:
    """sample_id -> score; samples a scorer marked not_measured / not_tested are left out."""
    d = json.loads((run_dir / f"{s}.json").read_text())
    out = {}
    for p in d["predictions"]:
        for sc in (p.get("metadata") or {}).get("scores", []):
            if sc.get("name") == METRIC[s] and not (sc.get("metadata") or {}).get("unknown"):
                out[str(p["sample_id"])] = float(sc["value"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    runs = sorted(p for p in Path(a.runs).iterdir() if (p / "summary.json").exists())
    ref = Path(a.runs) / a.ref
    table = {"reference": a.ref, "rows": []}
    lines = ["| Variant | Metric | Value | Δ vs " + a.ref + " [95% CI] | p |", "|---|---|---|---|---|"]
    for r in runs:
        for s in METRIC:
            if not (r / f"{s}.json").exists():
                continue
            cur = per_sample(r, s)
            val = sum(cur.values()) / max(1, len(cur))
            row = {"variant": r.name, "set": s, "metric": METRIC[s], "value": val, "n": len(cur)}
            if r != ref and (ref / f"{s}.json").exists():
                base = per_sample(ref, s)
                ids = sorted(set(base) & set(cur))
                bs = paired_bootstrap([(base[i], cur[i]) for i in ids], n_resamples=2000)
                row.update({"delta": bs.get("delta"), "ci": bs.get("ci"), "p": bs.get("p_value")})
                ci = bs.get("ci") or [float("nan")] * 2
                lines.append(f"| {r.name} | {LABEL[s]} | {val:.1%} (n={len(cur)}) | {bs['delta']:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] | {bs['p_value']:.3f} |")
            else:
                lines.append(f"| {r.name} | {LABEL[s]} | {val:.1%} (n={len(cur)}) | reference | |")
            table["rows"].append(row)
    Path(a.out, "results.json").write_text(json.dumps(table, indent=2))
    Path(a.out, "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
