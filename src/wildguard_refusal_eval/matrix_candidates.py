#!/usr/bin/env python3
"""Build a generic, system-prompt-sharded WildGuard candidate JSONL from a response matrix.

The input response matrix is deliberately treated as immutable external data.  This
module does not generate answers, score them, or infer their source policy; it only
turns already materialized ``(sample_id, system_prompt_id, response)`` records into
the compact candidate contract consumed by :mod:`wildguard_refusal_eval.judge`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = "wildguard_matrix_candidate_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_json_array(path: Path) -> Iterator[Any]:
    """Incrementally decode a top-level JSON array without loading a multi-GB file."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    need_separator = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            eof = not chunk
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} must contain a top-level JSON array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position >= len(buffer):
                    break
                if need_separator:
                    if buffer[position] == ",":
                        position += 1
                        need_separator = False
                        continue
                    if buffer[position] == "]":
                        finished = True
                        position += 1
                        while position < len(buffer) and buffer[position].isspace():
                            position += 1
                        if position != len(buffer) and not eof:
                            break
                        if position != len(buffer):
                            raise ValueError(f"Unexpected trailing content in {path}")
                        return
                    raise ValueError(f"Expected ',' or ']' in {path}")
                if buffer[position] == "]":
                    finished = True
                    position += 1
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position != len(buffer) and not eof:
                        break
                    if position != len(buffer):
                        raise ValueError(f"Unexpected trailing content in {path}")
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Malformed or truncated JSON array: {path}")
                    break
                yield value
                position = end
                need_separator = True
            if position:
                buffer = buffer[position:]
                position = 0
            if eof:
                if not finished:
                    raise ValueError(f"Missing closing ']' in {path}")
                return


def load_prompt_map(path: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for index, row in enumerate(iter_json_array(path), start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: prompt row {index} is not an object")
        sample_id = row.get("sample_id")
        prompt = row.get("prompt")
        if not isinstance(sample_id, int) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: invalid prompt row {index}")
        if sample_id in prompts:
            raise ValueError(f"{path}: duplicate sample_id={sample_id}")
        prompts[sample_id] = prompt
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def parser() -> argparse.ArgumentParser:
    parsed = argparse.ArgumentParser(description=__doc__)
    parsed.add_argument("--mode", choices=("plan", "run"), default="plan")
    parsed.add_argument("--responses", type=Path, required=True)
    parsed.add_argument("--prompts", type=Path, required=True)
    parsed.add_argument("--output-dir", type=Path, required=True)
    parsed.add_argument("--backbone", required=True, help="Stable label, e.g. base or instruct")
    parsed.add_argument("--response-field", default="filtered_response")
    parsed.add_argument("--system-prompt-begin-index", type=int, required=True)
    parsed.add_argument("--system-prompt-end-index-exclusive", type=int, required=True)
    parsed.add_argument("--expected-system-prompt-count", type=int, default=90)
    parsed.add_argument("--confirm-build", action="store_true")
    return parsed


def main() -> None:
    args = parser().parse_args()
    responses = args.responses.resolve()
    prompts_path = args.prompts.resolve()
    output_dir = args.output_dir.resolve()
    begin = args.system_prompt_begin_index
    end = args.system_prompt_end_index_exclusive
    if not responses.is_file() or not prompts_path.is_file():
        raise FileNotFoundError("Both --responses and --prompts must be existing files")
    if not (0 <= begin < end <= args.expected_system_prompt_count):
        raise ValueError("Require 0 <= begin < end <= expected system-prompt count")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "responses": str(responses),
        "responses_bytes": responses.stat().st_size,
        "prompts": str(prompts_path),
        "prompts_bytes": prompts_path.stat().st_size,
        "backbone": args.backbone,
        "response_field": args.response_field,
        "system_prompt_begin_index": begin,
        "system_prompt_end_index_exclusive": end,
        "expected_system_prompt_count": args.expected_system_prompt_count,
        "output_dir": str(output_dir),
        "output_file": str(output_dir / "candidates.jsonl"),
    }
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if not args.confirm_build:
        raise RuntimeError("Matrix candidate build requires --confirm-build")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    prompt_map = load_prompt_map(prompts_path)
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidates.jsonl"
    candidate_digest = hashlib.sha256()
    matrix_rows = selected_rows = 0
    seen_example_ids: set[str] = set()
    try:
        with candidate_path.open("x", encoding="utf-8", newline="\n") as output:
            for row_index, row in enumerate(iter_json_array(responses), start=1):
                matrix_rows += 1
                if not isinstance(row, dict):
                    raise ValueError(f"{responses}: response row {row_index} is not an object")
                sample_id = row.get("sample_id")
                arm_id = row.get("system_prompt_id")
                response = row.get(args.response_field)
                if not isinstance(sample_id, int) or not isinstance(arm_id, int):
                    raise ValueError(f"{responses}: invalid IDs at row {row_index}")
                if not 0 <= arm_id < args.expected_system_prompt_count:
                    raise ValueError(f"{responses}: arm_id={arm_id} outside expected range")
                if not begin <= arm_id < end:
                    continue
                if sample_id not in prompt_map:
                    raise ValueError(f"{responses}: missing prompt for sample_id={sample_id}")
                if not isinstance(response, str):
                    raise ValueError(f"{responses}: {args.response_field} is not a string at row {row_index}")
                example_id = f"{args.backbone}-q{sample_id:05d}-a{arm_id:03d}"
                if example_id in seen_example_ids:
                    raise ValueError(f"{responses}: duplicate matrix key {example_id}")
                seen_example_ids.add(example_id)
                candidate = {
                    "example_id": example_id,
                    "query_id": sample_id,
                    "arm_id": arm_id,
                    "prompt": prompt_map[sample_id],
                    "response": response,
                    "backbone": args.backbone,
                }
                encoded = (json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                output.buffer.write(encoded)
                candidate_digest.update(encoded)
                selected_rows += 1
        expected_rows = len(prompt_map) * (end - begin)
        if selected_rows != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} selected rows ({len(prompt_map)} prompts x {end - begin} arms), got {selected_rows}"
            )
        manifest = {
            **plan,
            "created_at_utc": utc_now(),
            "prompt_count": len(prompt_map),
            "response_matrix_rows": matrix_rows,
            "candidate_rows": selected_rows,
            "candidate_sha256": candidate_digest.hexdigest(),
            "responses_sha256": sha256_file(responses),
            "prompts_sha256": sha256_file(prompts_path),
        }
        (output_dir / "candidate_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "completed", **manifest}, ensure_ascii=False, indent=2))
    except Exception:
        # Keep incomplete bytes out of a falsely reusable output directory.
        candidate_path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise


if __name__ == "__main__":
    main()
