"""Evaluate one GGUF variant with AuditKIT through a local llama-server.

Every variant (bf16-equivalent Q8_0 reference, deployment Q4_K_M, repaired, ...)
is served the same way, so the only thing that changes between rows of the
results table is the weights.

Usage:
  python -m sns.evaluate --gguf models/gguf/global-Q4_K_M.gguf --name global-Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import auditkit as ak
from auditkit.metrics.refusal import refusal_scorer
from auditkit.runspec import RunConfig
from auditkit.types import Direction

PARALLEL = 8
MAX_TOKENS = {"harmful": 96, "benign": 96, "gsm8k": 320}


def _num(s: str):
    s = s.replace(",", "").replace("$", "")
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", s) or re.search(r"answer is\s*:?\s*\**\s*(-?\d+(?:\.\d+)?)", s, re.I)
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[-1]) if nums else None


@ak.scorer(direction=Direction.MAXIMIZE, name="gsm8k_acc")
def gsm8k_acc(sample, output):
    pred = _num(output or "")
    return 1.0 if pred is not None and abs(pred - float(sample.target)) < 1e-6 else 0.0


SCORERS = {
    "harmful": [refusal_scorer("refusal_rate", direction=Direction.MAXIMIZE)],
    "benign": [refusal_scorer("over_refusal", direction=Direction.MINIMIZE)],
    "gsm8k": [gsm8k_acc],
}


def load_samples(path: Path, limit: int | None):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    return [ak.Sample(input=r["prompt"], target=r.get("target", ""),
                      metadata={"id": r["id"], "source": r["source"], "category": r.get("category", "")})
            for r in rows]


def start_server(gguf: str, port: int, log: Path):
    cmd = ["llama-server", "-m", gguf, "--port", str(port), "-np", str(PARALLEL),
           "-c", str(PARALLEL * 1536), "-ngl", "99", "--jinja", "--temp", "0", "--seed", "0"]
    proc = subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited; see {log}")
        time.sleep(1)
    proc.kill()
    raise TimeoutError("llama-server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/eval")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--sets", default="harmful,benign,gsm8k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=8091)
    a = ap.parse_args()

    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    import os
    proc = None if os.environ.get("SNS_NO_SERVER") else start_server(a.gguf, a.port, out / "llama-server.log")
    summary = {"name": a.name, "gguf": a.gguf, "sets": {}}
    try:
        for s in a.sets.split(","):
            samples = load_samples(Path(a.data) / f"{s}.jsonl", a.limit)
            t0 = time.time()
            res = ak.evaluate(
                samples, model=f"api:{a.name}", scorers=SCORERS[s],
                api_base=f"http://127.0.0.1:{a.port}/v1", api_key="EMPTY",
                config=RunConfig(concurrency=PARALLEL, temperature=0.0, seed=0,
                                 max_tokens=MAX_TOKENS[s], timeout=600),
                cache=False, tags=[a.name, s],
            )
            dt = time.time() - t0
            res.save(str(out / f"{s}.json"))
            summary["sets"][s] = {"headline": res.headline, "n": len(samples), "seconds": round(dt, 1)}
            print(f"[{a.name}] {s}: {res.headline}  ({len(samples)} prompts, {dt:.0f}s)", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
