import pytest

torch = pytest.importorskip("torch")

from css700.models.cnn import SmallTabularImageCNN  # noqa: E402


def test_small_cnn_accepts_the_pilot_grid_and_exposes_grad_cam_layer() -> None:
    model = SmallTabularImageCNN(input_channels=1, num_classes=2)
    output = model(torch.zeros(4, 1, 5, 6))

    assert output.shape == (4, 2)
    assert model.grad_cam_layer is model.features[2]


@pytest.mark.parametrize("shape", [(6, 6), (8, 8)])
def test_same_cnn_architecture_handles_larger_synthetic_grids(shape) -> None:
    model = SmallTabularImageCNN(input_channels=1, num_classes=2)
    parameter_count = sum(p.numel() for p in model.parameters())
    inputs = torch.zeros(2, 1, *shape, requires_grad=True)
    output = model(inputs)
    assert output.shape == (2, 2)
    output.sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert sum(p.numel() for p in model.parameters()) == parameter_count
    assert model(torch.zeros(2, 1, 5, 6)).shape == output.shape
