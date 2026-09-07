from dataclasses import replace

import numpy as np
import pytest

from css700.encoding import GridEncoder
from css700.encoding.mapping import audit_feature_mapping
from css700.interpretability import trace_attributions_to_features


def test_tracing_uses_feature_identity_not_record_order() -> None:
    encoder = GridEncoder().fit(["a", "b", "c"])
    mapping = tuple(reversed(encoder.mapping or ()))
    values = np.array([[[[-2, 3], [5, 99]]]])
    np.testing.assert_array_equal(trace_attributions_to_features(values, mapping), [[-2, 3, 5]])
    assert audit_feature_mapping(mapping, (1, 2, 2))["empty_pixels"] == 1


def test_audit_retains_colliding_features_but_tracing_rejects_them() -> None:
    original = GridEncoder().fit(["a", "b", "c"]).mapping or ()
    mapping = (original[0], replace(original[1], column=0), original[2])
    audit = audit_feature_mapping(mapping, (1, 2, 2))
    assert audit["occupied_pixels"] == 2
    assert audit["features_in_collisions"] == 2
    assert audit["excess_assignments"] == 1
    assert audit["collisions"][0]["feature_names"] == ["a", "b"]
    with pytest.raises(ValueError, match="collisions"):
        trace_attributions_to_features(np.zeros((1, 1, 2, 2)), mapping)


@pytest.mark.parametrize(
    "change",
    [
        {"row": -1},
        {"column": 2},
        {"channel": -1},
        {"row": 0.5},
        {"feature_index": 1},
        {"feature_index": 0.0},
        {"feature_name": "b"},
        {"feature_name": ""},
    ],
)
def test_malformed_mapping_is_rejected(change: dict) -> None:
    mapping = GridEncoder().fit(["a", "b"]).mapping or ()
    with pytest.raises(ValueError):
        trace_attributions_to_features(
            np.zeros((1, 1, 1, 2)), (replace(mapping[0], **change), mapping[1])
        )


def test_empty_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        audit_feature_mapping([], (1, 2, 2))
