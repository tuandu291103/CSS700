import numpy as np
import pytest

from css700.encoding import GridEncoder
from css700.interpretability import trace_attributions_to_features


def test_grid_mapping_and_padding_are_explicit() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    encoder = GridEncoder(padding_value=-1.0)
    images = encoder.fit_transform(values, ["a", "b", "c", "d", "e"])

    assert images.shape == (1, 1, 2, 3)
    np.testing.assert_array_equal(images[0, 0], [[1.0, 2.0, 3.0], [4.0, 5.0, -1.0]])
    np.testing.assert_array_equal(
        encoder.padding_mask()[0], [[True, True, True], [True, True, False]]
    )
    assert encoder.mapping_records()[4]["feature_name"] == "e"
    assert encoder.mapping_records()[4]["row"] == 1
    assert encoder.mapping_records()[4]["column"] == 1


def test_attributions_trace_back_in_original_feature_order() -> None:
    encoder = GridEncoder().fit(["a", "b", "c"])
    attributions = np.array([[[[-2.0, 3.0], [5.0, 99.0]]]])

    signed = trace_attributions_to_features(attributions, encoder.mapping or ())
    magnitude = trace_attributions_to_features(attributions, encoder.mapping or (), absolute=True)
    np.testing.assert_array_equal(signed, [[-2.0, 3.0, 5.0]])
    np.testing.assert_array_equal(magnitude, [[2.0, 3.0, 5.0]])


def test_grid_rejects_duplicate_feature_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        GridEncoder().fit(["same", "same"])
