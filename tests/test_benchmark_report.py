import numpy as np

from wildguard_refusal_eval.benchmark_report import metric_row


def test_binary_metrics_and_miou() -> None:
    result = metric_row(np.array([False, False, True, True]), np.array([False, True, False, True]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["f1"] == 0.5
    assert result["mIoU"] == 1 / 3
