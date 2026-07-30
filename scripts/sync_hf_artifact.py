#!/usr/bin/env python3
"""Transfer immutable large evaluator artifacts through a Hugging Face dataset repo.

Authentication intentionally uses only the normal Hugging Face environment/cache;
this script has no token argument and never prints credentials.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("upload", "download"), required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-path", type=Path, required=True)
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--download-backend", choices=("hub", "requests"), default="hub")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise RuntimeError("Transfer requires --confirm")
    from huggingface_hub import HfApi, hf_hub_download

    local_path = args.local_path.resolve()
    if args.action == "upload":
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local_path), repo_id=args.repo_id, repo_type="dataset",
            path_in_repo=args.repo_path, revision=args.revision,
            commit_message=f"Add immutable WildGuard evaluator artifact: {args.repo_path}",
        )
        result = {"action": "upload", "repo_id": args.repo_id, "repo_path": args.repo_path,
                  "local_path": str(local_path), "bytes": local_path.stat().st_size,
                  "sha256": sha256_file(local_path)}
    else:
        if local_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing local path: {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if args.download_backend == "hub":
            downloaded = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.repo_path,
                                         revision=args.revision, local_dir=str(local_path.parent),
                                         local_dir_use_symlinks=False)
            downloaded_path = Path(downloaded)
            if downloaded_path.resolve() != local_path:
                downloaded_path.replace(local_path)
        else:
            # Avoid the Hub client's preliminary HEAD request when a constrained
            # network route stalls there. Requests still obtains the normal token
            # from Hugging Face's secure cache/environment and follows the Hub's
            # signed download redirect; it never prints or persists credentials.
            from huggingface_hub import get_token
            import requests

            token = get_token()
            if not token:
                raise RuntimeError("No Hugging Face credential available for authenticated download")
            url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/{args.revision}/{args.repo_path}"
            temporary = local_path.with_name(local_path.name + ".download-partial")
            try:
                with requests.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    stream=True,
                    timeout=(60.0, args.timeout_seconds),
                ) as response:
                    response.raise_for_status()
                    with temporary.open("xb") as handle:
                        for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                os.replace(temporary, local_path)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        result = {"action": "download", "repo_id": args.repo_id, "repo_path": args.repo_path,
                  "download_backend": args.download_backend, "local_path": str(local_path), "bytes": local_path.stat().st_size,
                  "sha256": sha256_file(local_path)}
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
