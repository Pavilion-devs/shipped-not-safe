"""Publish Round 1 artifacts to the Hugging Face Hub (needs `hf auth login` with a write token).

Creates:
  <user>/shipped-not-safe-eval   (dataset): curated eval sets, CuratorKIT provenance,
                                            raw AuditKIT runs (every generation), results table
  <user>/shipped-not-safe-recipes (model): quantization recipe, logs and sha256s
  two public collections (models / datasets) for the submission form
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    api = HfApi()
    user = api.whoami()["name"]
    ds = f"{user}/shipped-not-safe-eval"
    md = f"{user}/shipped-not-safe-recipes"
    api.create_repo(ds, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(repo_id=ds, repo_type="dataset", folder_path=str(ROOT / "data/eval"), path_in_repo="eval",
                      commit_message="Curated eval sets (CuratorKIT) + provenance")
    if any((ROOT / "results").rglob("*.json")):
        api.upload_folder(repo_id=ds, repo_type="dataset", folder_path=str(ROOT / "results"), path_in_repo="results",
                          commit_message="AuditKIT runs: every generation, per-sample scores, results table")
    api.upload_file(repo_id=ds, repo_type="dataset", path_or_fileobj=str(ROOT / "hub/DATASET_CARD.md"),
                    path_in_repo="README.md")

    api.create_repo(md, repo_type="model", exist_ok=True, private=False)
    for f in list((ROOT / "models/gguf").glob("*.quantize.log")) + [ROOT / "scripts/quantize.sh",
                                                                    ROOT / "scripts/run_round1.sh"]:
        if f.exists():
            api.upload_file(repo_id=md, path_or_fileobj=str(f), path_in_repo=f"recipe/{f.name}")
    api.upload_file(repo_id=md, path_or_fileobj=str(ROOT / "hub/MODEL_CARD.md"), path_in_repo="README.md")
    api.upload_folder(repo_id=md, folder_path=str(ROOT), path_in_repo="code",
                      allow_patterns=["sns/*.py", "scripts/*.sh", "README.md", "hub/*.md", ".gitignore"],
                      commit_message="Code mirror (also on GitHub)")

    mc = api.create_collection(title="Shipped != Safe: models", namespace=user, private=False, exists_ok=True,
                               description="Tiny Aya Global deployment quantizations audited for safety (Lexsi hackathon, Track 2a).")
    for rid in ["CohereLabs/tiny-aya-global-GGUF", "CohereLabs/tiny-aya-global", md]:
        try:
            api.add_collection_item(mc.slug, item_id=rid, item_type="model", exists_ok=True)
        except Exception as e:
            print("collection add failed", rid, e)
    dc = api.create_collection(title="Shipped != Safe: data", namespace=user, private=False, exists_ok=True,
                               description="Curated safety / over-refusal eval sets, logs and per-sample results.")
    api.add_collection_item(dc.slug, item_id=ds, item_type="dataset", exists_ok=True)
    print("MODEL COLLECTION:  https://huggingface.co/collections/" + mc.slug)
    print("DATASET COLLECTION: https://huggingface.co/collections/" + dc.slug)
    print("DATASET:", f"https://huggingface.co/datasets/{ds}")


if __name__ == "__main__":
    main()
