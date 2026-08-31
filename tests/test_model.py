import pytest

torch = pytest.importorskip("torch")

from css700.models.cnn import SmallTabularImageCNN  # noqa: E402


def test_small_cnn_accepts_the_pilot_grid_and_exposes_grad_cam_layer() -> None:
    model = SmallTabularImageCNN(input_channels=1, num_classes=2)
    output = model(torch.zeros(4, 1, 5, 6))

    assert output.shape == (4, 2)
    assert model.grad_cam_layer is model.features[2]
