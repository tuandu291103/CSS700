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


@pytest.mark.parametrize(
    "features,shape,padding", [(30, (5, 6), 0), (34, (6, 6), 2), (57, (8, 8), 7), (60, (8, 8), 4)]
)
def test_candidate_dimensions_preserve_values_and_exclude_padding(features, shape, padding):
    """Synthetic identities only: candidate feature counts do not select a dataset."""
    names = tuple(f"feature_{i}" for i in range(features - 2)) + ("char_freq_$", "char_freq_[")
    values = np.arange(2 * features, dtype=np.float32).reshape(2, features) - features
    encoder = GridEncoder(padding_value=-999.0)
    images = encoder.fit_transform(values, names)
    assert images.shape == (2, 1, *shape)
    mask = encoder.padding_mask()
    assert int((~mask).sum()) == padding
    np.testing.assert_array_equal(images[:, mask], values)
    assert np.all(images[:, ~mask] == -999.0)
    # Even very large padding attribution cannot acquire a named feature identity.
    maps = images.copy()
    maps[:, ~mask] = 1e9
    reordered = tuple(reversed(encoder.mapping or ()))
    np.testing.assert_array_equal(trace_attributions_to_features(maps, reordered), values)
    np.testing.assert_array_equal(
        trace_attributions_to_features(maps, reordered, absolute=True), np.abs(values)
    )
    assert [r["feature_name"] for r in encoder.mapping_records()] == list(names)
