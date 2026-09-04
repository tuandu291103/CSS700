import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from css700.interpretability import occlusion_sensitivity  # noqa: E402


class LinearImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 1.0, 2.0, -0.5]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(start_dim=1))


def _single_pixel_masks() -> torch.Tensor:
    return torch.eye(4, dtype=torch.bool).reshape(4, 1, 2, 2)


def test_occlusion_masks_exact_features_and_returns_signed_logit_drops() -> None:
    model = LinearImageClassifier()
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[0.5, -1.0], [2.0, 1.5]]]])
    targets = torch.tensor([0, 1])

    result = occlusion_sensitivity(
        model,
        inputs,
        feature_masks=_single_pixel_masks(),
        replacement_values=torch.zeros(1, 1, 2, 2),
        target_classes=targets,
        perturbation_batch_size=3,
    )

    expected = np.asarray([[1.0, -4.0, 1.5, 12.0], [-0.5, -1.0, 4.0, -0.75]])
    np.testing.assert_allclose(result.logit_drops.numpy(), expected, atol=1e-6)
    assert result.probability_drops.shape == (2, 4)
    assert torch.isfinite(result.probability_drops).all()
    assert torch.equal(result.target_classes, targets)
    assert result.batch_forward_calls == 4  # one original plus ceil(8 / 3) perturbation calls


def test_occlusion_defaults_to_predicted_class_and_validates_masks() -> None:
    model = LinearImageClassifier()
    inputs = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    result = occlusion_sensitivity(
        model,
        inputs,
        feature_masks=_single_pixel_masks(),
        replacement_values=torch.zeros_like(inputs),
    )

    assert torch.equal(result.target_classes, result.original_logits.argmax(dim=1))
    empty_masks = _single_pixel_masks()
    empty_masks[2] = False
    with pytest.raises(ValueError, match="every feature mask"):
        occlusion_sensitivity(
            model,
            inputs,
            feature_masks=empty_masks,
            replacement_values=torch.zeros_like(inputs),
        )
