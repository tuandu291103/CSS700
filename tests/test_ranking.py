import pytest

from css700.evaluation import spearman_agreement, top_k_overlap


def test_rank_metrics() -> None:
    assert spearman_agreement([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    overlap = top_k_overlap([5, 4, 1, 0], [5, 1, 4, 0], k=2)
    assert overlap["intersection"] == 1
    assert overlap["overlap_fraction"] == 0.5
    assert overlap["jaccard"] == pytest.approx(1 / 3)
