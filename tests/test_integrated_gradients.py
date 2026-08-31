import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("captum")
from torch import nn  # noqa: E402

from css700.encoding import GridEncoder  # noqa: E402
from css700.experiments.integrated_gradients import _build_baselines  # noqa: E402
from css700.interpretability import integrated_gradients  # noqa: E402


class LinearImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, -2.0, 0.5, 3.0], [-1.0, 1.0, 2.0, -0.5]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(start_dim=1))


def test_integrated_gradients_is_signed_targeted_and_complete() -> None:
    model = LinearImageClassifier()
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]], [[[0.5, -1.0], [2.0, 1.5]]]])
    targets = torch.tensor([0, 1])

    result = integrated_gradients(
        model,
        inputs,
        baselines=torch.zeros(1, 1, 2, 2),
        target_classes=targets,
        n_steps=32,
        internal_batch_size=16,
    )

    assert result.attributions.shape == inputs.shape
    assert torch.equal(result.target_classes, targets)
    assert torch.any(result.attributions < 0)
    np.testing.assert_allclose(
        result.attributions.sum(dim=(1, 2, 3)).numpy(),
        result.output_difference.numpy(),
        atol=1e-5,
    )
    assert torch.max(torch.abs(result.convergence_delta)) < 1e-5


def test_standardized_training_mean_is_deduplicated_from_zero() -> None:
    train_values = np.asarray([[-2.0, -4.0], [1.0, 2.0], [1.0, 2.0]])
    encoder = GridEncoder().fit(["a", "b"])

    baselines, aliases, diagnostics = _build_baselines(
        train_values,
        encoder,
        ["zero", "training_mean", "training_median"],
        equivalence_atol=1e-7,
    )

    assert aliases == {
        "zero": "zero",
        "training_mean": "zero",
        "training_median": "training_median",
    }
    assert set(baselines) == {"zero", "training_median"}
    assert diagnostics["training_mean"]["maximum_absolute_difference_from_zero"] == 0.0
    assert diagnostics["training_median"]["maximum_absolute_difference_from_zero"] == 2.0
