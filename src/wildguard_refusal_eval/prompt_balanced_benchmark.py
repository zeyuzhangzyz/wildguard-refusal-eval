#!/usr/bin/env python3
"""Prompt-risk-balanced, prompt-disjoint response-refusal holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from wildguard_refusal_eval.benchmark import PROXY_CONFIG, make_proxy, response_example, select_threshold, sha256_file
from wildguard_refusal_eval.benchmark_report import metric_row

HARM_LABELS = ("unharmful", "harmful")
PAIR_LABELS = {"compliance", "refusal"}
SELECTION_VERSION = "prompt_balanced_holdout_v1"


def digest(namespace: str, value: str) -> str:
    return hashlib.sha256((namespace + "\x00" + value).encode("utf-8")).hexdigest()


def load_rows(path: Path) -> Any:
    import pandas as pd

    table = pd.read_parquet(path, columns=["prompt", "response", "prompt_harm_label", "response_refusal_label"])
    table = table[table["prompt_harm_label"].isin(HARM_LABELS) & table["response_refusal_label"].isin(PAIR_LABELS)].copy()
    if table.empty:
        raise ValueError("No rows with both prompt-harm and response-refusal labels")
    return table.reset_index(drop=True)


def make_holdout(table: Any, pairs_per_harm: int) -> tuple[Any, Any, dict[str, Any]]:
    """Use 1,000 two-response prompts to produce 2,000 QA pairs per stratum."""
    if pairs_per_harm <= 0:
        raise ValueError("pairs_per_harm must be positive")
    selected: dict[str, list[str]] = {}
    details: dict[str, Any] = {}
    for harm in HARM_LABELS:
        subset = table[table["prompt_harm_label"] == harm]
        grouped = subset.groupby("prompt", sort=False)["response_refusal_label"].agg(list)
        eligible = [str(prompt) for prompt, labels in grouped.items() if len(labels) == 2 and set(map(str, labels)) == PAIR_LABELS]
        ordered = sorted(eligible, key=lambda prompt: digest(f"{SELECTION_VERSION}:{harm}", prompt))
        if len(ordered) < pairs_per_harm:
            raise ValueError(f"Need {pairs_per_harm} paired prompts for {harm}; only {len(ordered)} available")
        selected[harm] = ordered[:pairs_per_harm]
        details[harm] = {"eligible_unique_prompts": len(eligible), "selected_unique_prompts": pairs_per_harm, "selected_prompt_hash": digest(f"{SELECTION_VERSION}:{harm}:selected", "\n".join(selected[harm]))}
    masks = [(table["prompt_harm_label"] == harm) & table["prompt"].isin(prompts) for harm, prompts in selected.items()]
    held_mask = np.logical_or.reduce(masks)
    holdout, train = table[held_mask].copy(), table[~held_mask].copy()
    holdout["_prompt_hash"] = holdout.apply(lambda row: digest(f"{SELECTION_VERSION}:{row['prompt_harm_label']}", str(row["prompt"])), axis=1)
    holdout["_label_order"] = holdout["response_refusal_label"].map({"compliance": 0, "refusal": 1})
    holdout = holdout.sort_values(["prompt_harm_label", "_prompt_hash", "_label_order"]).drop(columns=["_prompt_hash", "_label_order"]).reset_index(drop=True)
    counts = holdout.groupby(["prompt_harm_label", "response_refusal_label"]).size().to_dict()
    for harm in HARM_LABELS:
        for response_label in PAIR_LABELS:
            if counts.get((harm, response_label), 0) != pairs_per_harm:
                raise AssertionError(f"Unexpected holdout labels: {counts}")
    return train.reset_index(drop=True), holdout, {"selection_version": SELECTION_VERSION, "unique_prompts_per_prompt_harm_label": pairs_per_harm, "qa_pairs_per_prompt_harm_label": pairs_per_harm * 2, "selection": details, "training_rows_after_prompt_holdout": int(len(train)), "per_harm_response_label_counts": {f"{harm}/{response}": int(value) for (harm, response), value in sorted(counts.items())}}


def validation_mask(table: Any) -> np.ndarray:
    prompts = [str(value) for value in table["prompt"].drop_duplicates().tolist()]
    ordered = sorted(prompts, key=lambda prompt: digest("prompt_balanced_threshold_validation", prompt))
    held = set(ordered[:max(1, round(len(ordered) * 0.2))])
    return table["prompt"].astype(str).isin(held).to_numpy()


def texts(table: Any) -> list[str]:
    return [response_example(prompt, response) for prompt, response in zip(table["prompt"], table["response"])]


def save(output_dir: Path, source: Path, holdout: Any, selection: dict[str, Any], threshold: dict[str, Any], metrics: list[dict[str, Any]], logistic_c: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "holdout_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row_id, row in holdout.iterrows():
            handle.write(json.dumps({"example_id": f"prompt-balanced-{row_id:05d}", "prompt_sha256": digest("holdout_prompt", str(row["prompt"])), "response_sha256": digest("holdout_response", str(row["response"])), "prompt_harm_label": str(row["prompt_harm_label"]), "response_refusal_label": str(row["response_refusal_label"])}, sort_keys=True) + "\n")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "task_type": "cpu_only", "scope": "Train-derived prompt-disjoint diagnostic, not WildGuardTest/external test. Exactly 2,000 QA pairs with unharmful requests and 2,000 with harmful requests.", "input_wildguard_train": str(source), "input_wildguard_train_sha256": sha256_file(source), "proxy_config": {**PROXY_CONFIG, "logistic_c": logistic_c}, "threshold_selection": threshold, "holdout_selection": selection, "metrics": metrics}
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    fields = ["split", "system", "n", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mIoU", "confusion_matrix"]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    lines = ["# Prompt-Risk-Balanced Response-Refusal Holdout", "", "Train-derived, prompt-disjoint diagnostic—not a replacement for official WildGuardTest. It contains exactly 2,000 QA pairs labelled `unharmful` and 2,000 labelled `harmful` at the request level. Each stratum contains 1,000 reference refusals and 1,000 compliances. The proxy was fit without these prompts; threshold selection used a separate prompt-disjoint validation subset of the remaining Train prompts.", "", f"Decision rule: `p(refusal) >= {threshold['reporting_threshold']:.2f}`; logistic `C={logistic_c:g}`.", "", "| Request-risk stratum | n | Accuracy | Balanced acc. | Precision | Recall | F1 | mIoU |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row['split']} | {row['n']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['mIoU']:.4f} |")
    lines += ["", "This evaluates binary response-refusal detection, not whether a refusal is appropriate."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildguard-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-harm", type=int, default=1000)
    parser.add_argument("--logistic-c", type=float, default=PROXY_CONFIG["logistic_c"])
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    args = parser.parse_args()
    if args.logistic_c <= 0:
        raise ValueError("--logistic-c must be positive")
    source, output_dir = args.wildguard_train.resolve(), args.output_dir.resolve()
    train, holdout, selection = make_holdout(load_rows(source), args.pairs_per_harm)
    val_mask = validation_mask(train)
    plan = {"mode": args.mode, "wildguard_train": str(source), "wildguard_train_sha256": sha256_file(source), "output_dir": str(output_dir), "logistic_c": args.logistic_c, "threshold_fitting_rows": int((~val_mask).sum()), "threshold_validation_rows": int(val_mask.sum()), "holdout_selection": selection, "scope": "Train-derived 4,000-QA-pair prompt-balanced holdout; official WildGuardTest remains the external test."}
    if args.mode == "plan":
        print(json.dumps(plan, indent=2)); return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    threshold_proxy = make_proxy(args.logistic_c)
    train_labels = train["response_refusal_label"].eq("refusal").astype(int).to_numpy()
    threshold_proxy.fit(texts(train.loc[~val_mask]), train_labels[~val_mask])
    threshold = select_threshold(threshold_proxy.predict_proba(texts(train.loc[val_mask]))[:, 1], train_labels[val_mask])
    proxy = make_proxy(args.logistic_c)
    proxy.fit(texts(train), train_labels)
    probabilities = proxy.predict_proba(texts(holdout))[:, 1]
    predicted = probabilities >= float(threshold["reporting_threshold"])
    metrics: list[dict[str, Any]] = []
    for split, mask in [("all", np.ones(len(holdout), dtype=bool))] + [(f"{harm}_request", holdout["prompt_harm_label"].eq(harm).to_numpy()) for harm in HARM_LABELS]:
        truth = holdout.loc[mask, "response_refusal_label"].eq("refusal").to_numpy()
        metrics.append({"split": split, "system": f"tfidf_proxy_p_ge_{float(threshold['reporting_threshold']):.2f}", **metric_row(truth, predicted[mask])})
    save(output_dir, source, holdout, selection, threshold, metrics, args.logistic_c)
    print(json.dumps({**plan, "threshold": threshold, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
