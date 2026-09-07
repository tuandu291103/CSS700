"""Validate feature identities and audit pixels before one-to-one tracing."""

from collections import defaultdict
from numbers import Integral
from typing import Any

from css700.encoding.grid import FeaturePixel


def audit_feature_mapping(
    mapping: tuple[FeaturePixel, ...] | list[FeaturePixel],
    image_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Reject malformed mappings but report collisions without resolving them.

    A shared pixel is not independently attributable to each contributing feature.
    Collision-free capacity alone does not guarantee a collision-free assignment.
    """

    if len(image_shape) != 3 or any(
        not isinstance(size, Integral) or isinstance(size, bool) or size <= 0
        for size in image_shape
    ):
        raise ValueError("image_shape must contain positive integer channels, rows, columns")
    if not mapping:
        raise ValueError("mapping must contain at least one feature")
    indices = [pixel.feature_index for pixel in mapping]
    if any(not isinstance(index, Integral) or isinstance(index, bool) for index in indices):
        raise ValueError("feature indices must be integers")
    if sorted(indices) != list(range(len(mapping))):
        raise ValueError("feature indices must cover 0 through feature_count - 1 exactly once")
    names = [pixel.feature_name for pixel in mapping]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("feature names must be nonempty and unique")

    groups: dict[tuple[int, int, int], list[FeaturePixel]] = defaultdict(list)
    for pixel in mapping:
        coordinates = (pixel.channel, pixel.row, pixel.column)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) or not 0 <= value < size
            for value, size in zip(coordinates, image_shape, strict=True)
        ):
            raise ValueError(f"mapping for {pixel.feature_name!r} is outside the image tensor")
        groups[coordinates].append(pixel)
    collisions = [
        {
            "channel": int(coordinates[0]),
            "row": int(coordinates[1]),
            "column": int(coordinates[2]),
            "feature_indices": [int(pixel.feature_index) for pixel in members],
            "feature_names": [pixel.feature_name for pixel in members],
        }
        for coordinates, members in sorted(groups.items())
        if len(members) > 1
    ]
    capacity = int(image_shape[0] * image_shape[1] * image_shape[2])
    return {
        "feature_count": len(mapping),
        "pixel_capacity": capacity,
        "occupied_pixels": len(groups),
        "empty_pixels": capacity - len(groups),
        "collision_pixel_count": len(collisions),
        "features_in_collisions": sum(len(group["feature_indices"]) for group in collisions),
        "excess_assignments": len(mapping) - len(groups),
        "one_to_one": not collisions,
        "collisions": collisions,
    }
