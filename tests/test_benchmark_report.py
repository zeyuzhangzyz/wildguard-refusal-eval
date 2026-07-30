import numpy as np

from wildguard_refusal_eval.benchmark import deterministic_stratified_validation_mask
from wildguard_refusal_eval.benchmark_report import metric_row


def test_binary_metrics_and_miou() -> None:
    result = metric_row(np.array([False, False, True, True]), np.array([False, True, False, True]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["f1"] == 0.5
    assert result["mIoU"] == 1 / 3


def test_deterministic_stratified_validation_mask() -> None:
    texts = [f"example-{index}" for index in range(20)]
    labels = np.array([0] * 10 + [1] * 10)
    first = deterministic_stratified_validation_mask(texts, labels)
    second = deterministic_stratified_validation_mask(texts, labels)
    assert np.array_equal(first, second)
    assert first.sum() == 4
    assert int(first[:10].sum()) == 2
    assert int(first[10:].sum()) == 2
