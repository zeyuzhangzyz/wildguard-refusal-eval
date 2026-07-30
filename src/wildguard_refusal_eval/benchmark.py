#!/usr/bin/env python3
"""Build a fixed WildGuardTest response-refusal benchmark candidate file."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORTING_THRESHOLD_DECIMALS = 1
PROXY_CONFIG = {
    "word_max_features": 60000,
    "char_max_features": 80000,
    "word_ngram_range": (1, 2),
    "char_ngram_range": (3, 5),
    "min_df": 3,
    "max_df": 0.995,
    "logistic_c": 2.0,
    "max_iter": 250,
    "seed": 20260729,
    "prompt_characters": 800,
    "response_characters": 1200,
    "reporting_threshold_decimals": REPORTING_THRESHOLD_DECIMALS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object, limit: int) -> str:
    return " ".join(str(value).split())[:limit]


def response_example(prompt: object, response: object) -> str:
    return (
        "[USER] " + clean_text(prompt, PROXY_CONFIG["prompt_characters"])
        + "\n[ASSISTANT] " + clean_text(response, PROXY_CONFIG["response_characters"])
    )


def load_labeled_rows(test_path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    table = pd.read_parquet(test_path, columns=["prompt", "response", "response_refusal_label"])
    table = table[table["response_refusal_label"].notna()].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row_id, row in table.iterrows():
        label = str(row["response_refusal_label"])
        if label not in {"refusal", "compliance"}:
            raise ValueError(f"Unexpected response_refusal_label={label!r} at row {row_id}")
        prompt = str(row["prompt"])
        response = str(row["response"])
        text = response_example(prompt, response)
        rows.append({
            "example_id": f"wildguardtest-{row_id:05d}",
            "benchmark_row_id": int(row_id),
            "prompt": prompt,
            "response": response,
            "ground_truth_refusal": label == "refusal",
            "split": "full_test",
        })
    if not rows:
        raise ValueError("WildGuardTest has no response-refusal labels")
    return rows


def make_proxy() -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=PROXY_CONFIG["word_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["word_max_features"], sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=PROXY_CONFIG["char_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["char_max_features"], sublinear_tf=True)),
    ])
    classifier = LogisticRegression(C=PROXY_CONFIG["logistic_c"], max_iter=PROXY_CONFIG["max_iter"], class_weight="balanced", solver="liblinear", random_state=PROXY_CONFIG["seed"])
    return Pipeline([("features", features), ("classifier", classifier)])


def binary_f1(labels: Any, predictions: Any) -> float:
    import numpy as np

    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def select_threshold(probabilities: Any, labels: Any) -> dict[str, float | int]:
    """Select an F1 operating threshold from full WildGuardTrain predictions."""
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    raw_threshold = float(thresholds[best])
    reporting_threshold = round(raw_threshold, REPORTING_THRESHOLD_DECIMALS)
    return {
        "raw_f1_optimal_threshold": raw_threshold,
        "raw_validation_f1": float(f1[best]),
        "reporting_threshold": reporting_threshold,
        "reporting_validation_f1": binary_f1(labels, np.asarray(probabilities) >= reporting_threshold),
    }


def fit_proxy_and_score(train_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import pandas as pd

    train = pd.read_parquet(train_path, columns=["prompt", "response", "response_refusal_label"])
    train = train[train["response_refusal_label"].notna()].reset_index(drop=True)
    texts = [response_example(prompt, response) for prompt, response in zip(train["prompt"], train["response"])]
    labels = train["response_refusal_label"].eq("refusal").astype(int).to_numpy()
    proxy = make_proxy()
    proxy.fit(texts, labels)
    threshold = select_threshold(proxy.predict_proba(texts)[:, 1], labels)
    probabilities = proxy.predict_proba([response_example(row["prompt"], row["response"]) for row in rows])[:, 1]
    for row, probability in zip(rows, probabilities):
        row["tfidf_refusal_probability"] = float(probability)
        row["tfidf_refusal_threshold"] = threshold["reporting_threshold"]
        row["tfidf_refusal_at_threshold"] = bool(probability >= threshold["reporting_threshold"])
    return {
        "source": "in-sample probability predictions from the final full WildGuardTrain fit",
        "calibration_records": len(texts),
        "fitting_records": len(texts),
        **threshold,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "build"), default="plan")
    parser.add_argument("--wildguard-train", type=Path, required=True)
    parser.add_argument("--wildguard-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train_path, test_path, output_dir = args.wildguard_train.resolve(), args.wildguard_test.resolve(), args.output_dir.resolve()
    rows = load_labeled_rows(test_path)
    plan = {
        "mode": args.mode,
        "wildguard_train": str(train_path),
        "wildguard_train_sha256": sha256_file(train_path),
        "wildguard_test": str(test_path),
        "wildguard_test_sha256": sha256_file(test_path),
        "records": len(rows),
        "refusal_positive_count": sum(row["ground_truth_refusal"] for row in rows),
        "test_protocol": "all labeled WildGuardTest rows are reserved for final evaluation only",
        "threshold_selection": "fit on all WildGuardTrain rows, select the F1-optimal in-sample training threshold, round to one reportable decimal, then score the full WildGuardTest once",
        "output_dir": str(output_dir),
    }
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    threshold = fit_proxy_and_score(train_path, rows)
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidates.jsonl"
    write_jsonl(candidate_path, rows)
    manifest = {**plan, "created_at_utc": utc_now(), "candidate_file": str(candidate_path), "candidate_sha256": sha256_file(candidate_path), "proxy_config": PROXY_CONFIG, "threshold": threshold}
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
