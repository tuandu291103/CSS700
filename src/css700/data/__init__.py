"""Dataset loading and leakage-safe preprocessing."""

from css700.data.breast_cancer import DataBundle, load_breast_cancer_splits

__all__ = ["DataBundle", "load_breast_cancer_splits"]
