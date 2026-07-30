#!/usr/bin/env python3
"""Report the fixed-threshold TF-IDF proxy on official WildGuardTest labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildguard_refusal_eval.benchmark_report import metric_row


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.candidates.resolve())
    if not rows:
        raise ValueError("Candidate file is empty")
    for row in rows:
        if not isinstance(row.get("ground_truth_refusal"), bool) or not isinstance(row.get("tfidf_refusal_probability"), float) or not isinstance(row.get("tfidf_refusal_threshold"), float):
            raise ValueError(f"{row.get('example_id')}: missing ground-truth or thresholded TF-IDF label")
    thresholds = {float(row["tfidf_refusal_threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError(f"Expected exactly one candidate threshold, got {sorted(thresholds)}")
    threshold = thresholds.pop()
    summaries: list[dict[str, Any]] = []
    y_true = np.asarray([row["ground_truth_refusal"] for row in rows], dtype=bool)
    predicted = np.asarray([row["tfidf_refusal_probability"] >= threshold for row in rows], dtype=bool)
    summaries.append({"split": "all", "system": f"tfidf_proxy_p_ge_{threshold:.2f}", **metric_row(y_true, predicted)})
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_split": "all",
        "threshold": threshold,
        "records": len(rows),
        "metrics": summaries,
        "scope": "All labeled WildGuardTest rows are used only for final evaluation. The reporting threshold was selected from in-sample predictions of the final full WildGuardTrain fit.",
    }
    (output_dir / "proxy_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["split", "system", "n", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mIoU", "confusion_matrix"]
    with (output_dir / "proxy_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# Train-Calibrated TF-IDF Refusal Proxy", "", f"Ground truth: official WildGuardTest `response_refusal_label`. All labeled test rows are final evaluation only. Threshold selected from in-sample predictions of the final full WildGuardTrain fit and rounded for reporting: `p >= {threshold:.2f}`.", "", "| Split | n | Accuracy | Balanced acc. | Precision | Recall | F1 | mIoU |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['split']} | {row['n']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['mIoU']:.4f} |")
    lines += ["", "This is binary response-refusal evaluation, not human full/partial refusal evaluation."]
    (output_dir / "proxy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "primary_split": "all", "records": len(rows), "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
