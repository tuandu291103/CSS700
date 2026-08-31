"""A fixed, one-feature-per-pixel Grid Encoding baseline."""

from dataclasses import asdict, dataclass
from math import ceil, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class FeaturePixel:
    """External mapping contract shared by all future encoders."""

    feature_index: int
    feature_name: str
    row: int
    column: int
    channel: int = 0
    mapping_version: str = "grid-v1"


class GridEncoder:
    """Place features row-by-row in a deterministic near-square grid."""

    def __init__(
        self,
        *,
        rows: int | None = None,
        columns: int | None = None,
        padding_value: float = 0.0,
    ) -> None:
        if (rows is None) != (columns is None):
            raise ValueError("rows and columns must either both be set or both be omitted")
        self.requested_rows = rows
        self.requested_columns = columns
        self.padding_value = float(padding_value)
        self.rows: int | None = None
        self.columns: int | None = None
        self.feature_names: tuple[str, ...] | None = None
        self.mapping: tuple[FeaturePixel, ...] | None = None

    def fit(self, feature_names: list[str] | tuple[str, ...]) -> "GridEncoder":
        names = tuple(str(name) for name in feature_names)
        if not names:
            raise ValueError("at least one feature is required")
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")

        if self.requested_rows is None:
            columns = ceil(sqrt(len(names)))
            rows = ceil(len(names) / columns)
        else:
            rows = int(self.requested_rows)
            columns = int(self.requested_columns)  # type: ignore[arg-type]

        if rows <= 0 or columns <= 0 or rows * columns < len(names):
            raise ValueError("grid dimensions must contain every feature")

        self.rows = rows
        self.columns = columns
        self.feature_names = names
        self.mapping = tuple(
            FeaturePixel(
                feature_index=index,
                feature_name=name,
                row=index // columns,
                column=index % columns,
            )
            for index, name in enumerate(names)
        )
        return self

    def transform(self, values: ArrayLike) -> NDArray[np.float32]:
        if self.mapping is None or self.rows is None or self.columns is None:
            raise RuntimeError("fit must be called before transform")
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("values must have shape [samples, features]")
        if array.shape[1] != len(self.mapping):
            raise ValueError(f"expected {len(self.mapping)} features, received {array.shape[1]}")

        images = np.full(
            (array.shape[0], 1, self.rows, self.columns),
            self.padding_value,
            dtype=np.float32,
        )
        for pixel in self.mapping:
            images[:, pixel.channel, pixel.row, pixel.column] = array[:, pixel.feature_index]
        return images

    def fit_transform(
        self,
        values: ArrayLike,
        feature_names: list[str] | tuple[str, ...],
    ) -> NDArray[np.float32]:
        return self.fit(feature_names).transform(values)

    def padding_mask(self) -> NDArray[np.bool_]:
        """Return True for occupied feature pixels and False for padding."""

        if self.mapping is None or self.rows is None or self.columns is None:
            raise RuntimeError("fit must be called before requesting a padding mask")
        mask = np.zeros((1, self.rows, self.columns), dtype=np.bool_)
        for pixel in self.mapping:
            mask[pixel.channel, pixel.row, pixel.column] = True
        return mask

    def mapping_records(self) -> list[dict[str, int | str]]:
        if self.mapping is None:
            raise RuntimeError("fit must be called before requesting mapping records")
        return [asdict(pixel) for pixel in self.mapping]
