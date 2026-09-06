import csv

import numpy as np
import pytest

pytest.importorskip("torch")
from css700.experiments.stability import _explanation_summaries, _feature_matrix


def test_feature_matrix_filters_and_restores_sample_feature_order(tmp_path) -> None:
    path = tmp_path / "features.csv"
    rows = [
        {"sample_id": 20, "feature_index": 1, "baseline": "zero", "value": -4.0},
        {"sample_id": 10, "feature_index": 0, "baseline": "zero", "value": 1.0},
        {"sample_id": 20, "feature_index": 0, "baseline": "zero", "value": 3.0},
        {"sample_id": 10, "feature_index": 1, "baseline": "zero", "value": -2.0},
        {"sample_id": 10, "feature_index": 0, "baseline": "other", "value": 99.0},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matrix = _feature_matrix(
        path,
        sample_ids=np.asarray([10, 20]),
        feature_count=2,
        value_field="value",
        row_filter=lambda row: row["baseline"] == "zero",
        absolute=True,
    )
    np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])


def test_explanation_summary_separates_same_and_different_targets() -> None:
    rows = [
        {
            "method": "grad_cam",
            "same_predicted_target": 1,
            "spearman_defined": 1,
            "spearman": 0.8,
            "top_5_overlap_fraction": 0.6,
        },
        {
            "method": "grad_cam",
            "same_predicted_target": 0,
            "spearman_defined": 1,
            "spearman": -0.2,
            "top_5_overlap_fraction": 0.2,
        },
    ]
    summary_rows, nested = _explanation_summaries(rows, ["grad_cam"], [5])

    assert len(summary_rows) == 3
    assert nested["grad_cam__same_predicted_target"]["median_spearman"] == 0.8
    assert nested["grad_cam__different_predicted_target"]["median_spearman"] == -0.2
