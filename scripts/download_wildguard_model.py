#!/usr/bin/env python3
"""Download the access-approved official WildGuard checkpoint with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts):
        entries.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run", "status"), default="plan")
    parser.add_argument("--model-id", default="allenai/wildguard")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-download", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "download_manifest.json"
    plan = {"mode": args.mode, "model_id": args.model_id, "revision": args.revision,
            "output_dir": str(output_dir), "claim_scope": "Weight preparation only; no model inference."}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2)); return
    if args.mode == "status":
        print(json.dumps({**plan, "exists": output_dir.exists(), "manifest_exists": manifest_path.exists()}, indent=2)); return
    if not args.confirm_download:
        raise RuntimeError("Model download requires --confirm-download")
    if manifest_path.exists():
        raise FileExistsError(f"Completed model output already exists: {output_dir}")
    from huggingface_hub import snapshot_download
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.model_id, revision=args.revision, local_dir=str(output_dir))
    files = inventory(output_dir)
    if not files:
        raise RuntimeError("Downloaded model directory is empty")
    manifest = {**plan, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": files, "total_bytes": sum(int(item["bytes"]) for item in files),
                "hf_token_env_present": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"))}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(output_dir),
                      "file_count": len(files), "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
