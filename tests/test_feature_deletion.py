import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from css700.evaluation import normalized_area_over_curve  # noqa: E402
from css700.interpretability import cumulative_feature_deletion  # noqa: E402


class LinearImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(start_dim=1))


def _single_pixel_masks() -> torch.Tensor:
    return torch.eye(4, dtype=torch.bool).reshape(4, 1, 2, 2)


def test_cumulative_deletion_follows_exact_ranked_feature_sets() -> None:
    model = LinearImageClassifier()
    inputs = torch.ones(2, 1, 2, 2)
    rankings = torch.tensor([[3, 2, 1, 0], [0, 1, 2, 3]])

    result = cumulative_feature_deletion(
        model,
        inputs,
        rankings=rankings,
        feature_masks=_single_pixel_masks(),
        replacement_values=torch.zeros(1, 1, 2, 2),
        k_values=[0, 1, 2, 4],
        target_classes=torch.tensor([0, 0]),
        inference_batch_size=8,
    )

    expected_drops = np.asarray([[0.0, 4.0, 7.0, 10.0], [0.0, 1.0, 3.0, 10.0]])
    np.testing.assert_allclose(result.target_logit_drops[0].numpy().T, expected_drops)
    np.testing.assert_array_equal(result.deleted_feature_counts[0, :, 0], [0, 1, 2, 4])
    assert result.predictions.shape == (1, 4, 2)
    assert result.batch_forward_calls == 4
    assert result.conceptual_model_evaluations == 8


def test_deletion_accepts_repeated_rankings_and_rejects_non_permutations() -> None:
    model = LinearImageClassifier()
    inputs = torch.ones(2, 1, 2, 2)
    repeated_rankings = torch.tensor([[[0, 1, 2, 3], [1, 0, 3, 2]], [[3, 2, 1, 0], [2, 3, 0, 1]]])

    result = cumulative_feature_deletion(
        model,
        inputs,
        rankings=repeated_rankings,
        feature_masks=_single_pixel_masks(),
        replacement_values=torch.zeros_like(inputs),
        k_values=[0, 2, 4],
    )
    assert result.target_logit_drops.shape == (2, 3, 2)
    np.testing.assert_allclose(result.target_logit_drops[:, -1].numpy(), np.full((2, 2), 10.0))

    bad_rankings = repeated_rankings.clone()
    bad_rankings[0, 0, -1] = 0
    with pytest.raises(ValueError, match="permutation"):
        cumulative_feature_deletion(
            model,
            inputs,
            rankings=bad_rankings,
            feature_masks=_single_pixel_masks(),
            replacement_values=torch.zeros_like(inputs),
            k_values=[0, 4],
        )


def test_normalized_area_over_curve_handles_irregular_steps() -> None:
    curves = np.asarray([[0.0, 2.0, 4.0], [0.0, 1.0, 2.0]])
    result = normalized_area_over_curve(curves, [0, 1, 4], feature_count=4, axis=1)
    np.testing.assert_allclose(result, [2.5, 1.25])
