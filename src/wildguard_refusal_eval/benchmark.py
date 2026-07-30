#!/usr/bin/env python3
"""Build a fixed WildGuardTest response-refusal benchmark candidate file."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_VALIDATION_FRACTION = 0.20
REPORTING_THRESHOLD_DECIMALS = 1
PROXY_CONFIG = {
    "word_max_features": 60000,
    "char_max_features": 80000,
    "word_ngram_range": (1, 2),
    "char_ngram_range": (3, 5),
    "min_df": 3,
    "max_df": 0.995,
    "logistic_c": 10.0,
    "max_iter": 250,
    "seed": 20260729,
    "prompt_characters": 800,
    "response_characters": 1200,
    "threshold_validation_fraction": TRAIN_VALIDATION_FRACTION,
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


def deterministic_stratified_validation_mask(texts: list[str], labels: Any) -> Any:
    """Take a fixed approximately 20% validation partition within each label."""
    import numpy as np

    labels = np.asarray(labels)
    mask = np.zeros(len(texts), dtype=bool)
    for label in sorted(set(labels.tolist())):
        indices = [index for index, value in enumerate(labels) if value == label]
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                ("threshold_validation:" + texts[index]).encode("utf-8")
            ).digest(),
        )
        validation_count = max(1, round(len(ordered) * TRAIN_VALIDATION_FRACTION))
        mask[ordered[:validation_count]] = True
    return mask


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


def make_proxy(logistic_c: float) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=PROXY_CONFIG["word_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["word_max_features"], sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=PROXY_CONFIG["char_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["char_max_features"], sublinear_tf=True)),
    ])
    classifier = LogisticRegression(C=logistic_c, max_iter=PROXY_CONFIG["max_iter"], class_weight="balanced", solver="liblinear", random_state=PROXY_CONFIG["seed"])
    return Pipeline([("features", features), ("classifier", classifier)])


def make_classifier(logistic_c: float) -> Any:
    """Create the proxy's classifier independently of its fixed text features."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=logistic_c,
        max_iter=PROXY_CONFIG["max_iter"],
        class_weight="balanced",
        solver="liblinear",
        random_state=PROXY_CONFIG["seed"],
    )


def binary_f1(labels: Any, predictions: Any) -> float:
    import numpy as np

    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def select_threshold(probabilities: Any, labels: Any) -> dict[str, float | int]:
    """Select an F1 operating threshold from Train validation predictions."""
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


def load_train_texts_labels(train_path: Path) -> tuple[list[str], Any]:
    import pandas as pd

    train = pd.read_parquet(train_path, columns=["prompt", "response", "response_refusal_label"])
    train = train[train["response_refusal_label"].notna()].reset_index(drop=True)
    texts = [response_example(prompt, response) for prompt, response in zip(train["prompt"], train["response"])]
    labels = train["response_refusal_label"].eq("refusal").astype(int).to_numpy()
    return texts, labels


def tune_logistic_c_on_train_validation(train_path: Path, logistic_cs: list[float]) -> dict[str, Any]:
    """Choose C solely from the fixed held-out WildGuardTrain validation fold.

    The TF-IDF vocabulary and training-fold matrix are intentionally built once.
    They are independent of C, so this is equivalent to fitting one pipeline per
    candidate while avoiding repeated expensive vectorization.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    if not logistic_cs or any(value <= 0 for value in logistic_cs):
        raise ValueError("Every logistic C candidate must be positive")
    texts, labels = load_train_texts_labels(train_path)
    validation_mask = deterministic_stratified_validation_mask(texts, labels)
    fitting_texts = [text for text, keep in zip(texts, ~validation_mask) if keep]
    validation_texts = [text for text, keep in zip(texts, validation_mask) if keep]
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=PROXY_CONFIG["word_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["word_max_features"], sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=PROXY_CONFIG["char_ngram_range"], min_df=PROXY_CONFIG["min_df"], max_df=PROXY_CONFIG["max_df"], max_features=PROXY_CONFIG["char_max_features"], sublinear_tf=True)),
    ])
    fitting_features = features.fit_transform(fitting_texts)
    validation_features = features.transform(validation_texts)
    rows: list[dict[str, Any]] = []
    for logistic_c in sorted(set(logistic_cs)):
        classifier = make_classifier(logistic_c)
        classifier.fit(fitting_features, labels[~validation_mask])
        probabilities = classifier.predict_proba(validation_features)[:, 1]
        threshold = select_threshold(probabilities, labels[validation_mask])
        rows.append({"logistic_c": logistic_c, **threshold})
    best = max(rows, key=lambda row: (float(row["reporting_validation_f1"]), float(row["raw_validation_f1"]), -float(row["logistic_c"])))
    return {
        "source": "deterministic stratified held-out validation partition of WildGuardTrain",
        "fitting_records": int((~validation_mask).sum()),
        "validation_records": int(validation_mask.sum()),
        "candidates": rows,
        "selected": best,
    }


def fit_proxy_and_score(train_path: Path, rows: list[dict[str, Any]], logistic_c: float) -> dict[str, Any]:
    texts, labels = load_train_texts_labels(train_path)
    validation_mask = deterministic_stratified_validation_mask(texts, labels)
    calibration_proxy = make_proxy(logistic_c)
    calibration_proxy.fit([text for text, keep in zip(texts, ~validation_mask) if keep], labels[~validation_mask])
    validation_probabilities = calibration_proxy.predict_proba([text for text, keep in zip(texts, validation_mask) if keep])[:, 1]
    threshold = select_threshold(validation_probabilities, labels[validation_mask])
    proxy = make_proxy(logistic_c)
    proxy.fit(texts, labels)
    probabilities = proxy.predict_proba([response_example(row["prompt"], row["response"]) for row in rows])[:, 1]
    for row, probability in zip(rows, probabilities):
        row["tfidf_refusal_probability"] = float(probability)
        row["tfidf_refusal_threshold"] = threshold["reporting_threshold"]
        row["tfidf_refusal_at_threshold"] = bool(probability >= threshold["reporting_threshold"])
    return {
        "source": "deterministic stratified held-out validation partition of WildGuardTrain",
        "validation_records": int(validation_mask.sum()),
        "fitting_records": int((~validation_mask).sum()),
        **threshold,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "build", "tune-plan", "tune"), default="plan")
    parser.add_argument("--wildguard-train", type=Path, required=True)
    parser.add_argument("--wildguard-test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logistic-c", type=float, default=PROXY_CONFIG["logistic_c"])
    parser.add_argument("--tune-logistic-cs", type=float, nargs="+", default=None)
    args = parser.parse_args()
    train_path, output_dir = args.wildguard_train.resolve(), args.output_dir.resolve()
    if args.logistic_c <= 0:
        raise ValueError("--logistic-c must be positive")
    if args.mode == "tune-plan":
        candidates = args.tune_logistic_cs or [args.logistic_c]
        if not candidates or any(value <= 0 for value in candidates):
            raise ValueError("Every --tune-logistic-cs value must be positive")
        print(json.dumps({
            "mode": "tune-plan",
            "wildguard_train": str(train_path),
            "wildguard_train_sha256": sha256_file(train_path),
            "test_protocol": "WildGuardTest is not loaded or inspected during hyperparameter selection",
            "output_dir": str(output_dir),
            "tune_logistic_cs": candidates,
        }, ensure_ascii=False, indent=2))
        return
    if args.mode == "tune":
        tune = tune_logistic_c_on_train_validation(train_path, args.tune_logistic_cs or [args.logistic_c])
        tune_plan = {
            "mode": "tune",
            "wildguard_train": str(train_path),
            "wildguard_train_sha256": sha256_file(train_path),
            "test_protocol": "WildGuardTest is not loaded or inspected during hyperparameter selection",
            "output_dir": str(output_dir),
            "tuning": tune,
        }
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        output_dir.mkdir(parents=True)
        (output_dir / "tuning_manifest.json").write_text(json.dumps(tune_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(tune_plan, ensure_ascii=False, indent=2))
        return
    if args.wildguard_test is None:
        raise ValueError("--wildguard-test is required unless --mode tune")
    test_path = args.wildguard_test.resolve()
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
        "threshold_selection": "fit on a deterministic stratified WildGuardTrain fitting partition, select the F1-optimal threshold on the held-out Train validation partition, round to one reportable decimal, refit on all WildGuardTrain rows, then score the full WildGuardTest once",
        "logistic_c": args.logistic_c,
        "output_dir": str(output_dir),
    }
    if args.mode == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    threshold = fit_proxy_and_score(train_path, rows, args.logistic_c)
    output_dir.mkdir(parents=True)
    candidate_path = output_dir / "candidates.jsonl"
    write_jsonl(candidate_path, rows)
    proxy_config = {**PROXY_CONFIG, "logistic_c": args.logistic_c}
    manifest = {**plan, "created_at_utc": utc_now(), "candidate_file": str(candidate_path), "candidate_sha256": sha256_file(candidate_path), "proxy_config": proxy_config, "threshold": threshold}
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
