import numpy as np
import pytest

torch = pytest.importorskip("torch")

from css700.interpretability import grad_cam  # noqa: E402
from css700.models import SmallTabularImageCNN  # noqa: E402


def test_grad_cam_is_class_targeted_finite_and_input_sized() -> None:
    torch.manual_seed(7)
    model = SmallTabularImageCNN(input_channels=1, num_classes=2)
    inputs = torch.randn(4, 1, 5, 6)
    targets = torch.tensor([0, 1, 0, 1])

    result = grad_cam(model, inputs, target_layer=model.grad_cam_layer, target_classes=targets)

    assert result.raw.shape == inputs.shape
    assert result.normalized.shape == inputs.shape
    assert torch.equal(result.target_classes, targets)
    assert torch.isfinite(result.raw).all()
    assert torch.all(result.raw >= 0)
    assert torch.all(result.normalized >= 0)
    assert torch.all(result.normalized <= 1)
    np.testing.assert_array_equal(result.logits.argmax(dim=1).shape, (4,))


def test_grad_cam_defaults_to_predicted_class_and_removes_hook() -> None:
    model = SmallTabularImageCNN(input_channels=1, num_classes=2)
    inputs = torch.randn(2, 1, 5, 6)
    hooks_before = len(model.grad_cam_layer._forward_hooks)

    result = grad_cam(model, inputs, target_layer=model.grad_cam_layer)

    assert torch.equal(result.target_classes, result.logits.argmax(dim=1))
    assert len(model.grad_cam_layer._forward_hooks) == hooks_before
