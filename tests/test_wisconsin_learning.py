"""Synthetic encoding/preprocessing checks; no downloaded data needed."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from scripts.wisconsin_learning import (
    FEATURES,
    decode,
    encode,
    load_data,
    make_mapping,
    split_rows,
)


class WisconsinLearningTests(unittest.TestCase):
    def test_feature_order_matches_documented_blocks(self):
        self.assertEqual(len(FEATURES), 30)
        self.assertEqual((FEATURES[0], FEATURES[10], FEATURES[20]),
                         ("radius_mean", "radius_se", "radius_worst"))

    def test_all_features_round_trip_exactly(self):
        values = np.arange(90, dtype=float).reshape(3, 30) / 7
        mapping = make_mapping()
        images, occupied = encode(values, mapping)
        np.testing.assert_array_equal(decode(images, mapping), values)
        self.assertEqual(images.shape, (3, 5, 6))
        self.assertTrue(occupied.all())

    def test_each_feature_only_changes_its_assigned_pixel(self):
        mapping = make_mapping()
        for item in mapping:
            values = np.zeros((1, 30))
            values[0, item["feature_index"]] = 1
            images, _ = encode(values, mapping)
            self.assertEqual(images.sum(), 1)
            self.assertEqual(images[0, item["row"], item["column"]], 1)

    def test_padding_is_separate_from_valid_zero(self):
        mapping = make_mapping(("a", "b", "c"), (2, 2))
        images, occupied = encode([[0, -1, 3]], mapping, (2, 2))
        self.assertTrue(occupied[0, 0])
        self.assertFalse(occupied[1, 1])
        np.testing.assert_array_equal(decode(images, mapping), [[0, -1, 3]])

    def test_rejects_collision_invalid_order_and_bounds(self):
        for field, value in (("row", 9), ("feature_index", 7), ("column", 0)):
            mapping = copy.deepcopy(make_mapping())
            mapping[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                encode(np.zeros((2, 30)), mapping)

    def test_rejects_invalid_shape_and_nonfinite_values(self):
        for values in (np.zeros((2, 29)), np.full((1, 30), np.nan), np.zeros(30)):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                encode(values, make_mapping())
        with self.assertRaises(ValueError):
            make_mapping(FEATURES, (2, 2))
        with self.assertRaises(ValueError):
            make_mapping(("a", "a"))

    def test_training_scaling_not_changed_by_holdout_extreme(self):
        scaler = MinMaxScaler(clip=False).fit([[2, 4], [4, 8]])
        before = scaler.data_max_.copy()
        transformed = scaler.transform([[20, 80]])
        self.assertTrue((transformed > 1).all())
        np.testing.assert_array_equal(scaler.data_max_, before)
        np.testing.assert_allclose(scaler.inverse_transform(transformed), [[20, 80]])

    def test_split_is_deterministic_disjoint_and_complete(self):
        labels = np.array(["B"] * 357 + ["M"] * 212)
        first, second = split_rows(labels), split_rows(labels)
        self.assertEqual([len(v) for v in first.values()], [397, 86, 86])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(first.values()))), np.arange(569))
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertLess(abs(np.mean(labels[first[name]] == "M") - 212 / 569), 0.02)

    def test_parser_excludes_id_and_label_from_predictors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.data"
            path.write_text("001,B," + ",".join(map(str, range(30))) + "\n")
            frame = load_data(path)
            self.assertEqual(frame.sample_id.iloc[0], "001")
            self.assertEqual(frame.loc[:, FEATURES].shape, (1, 30))

    def test_parser_rejects_missing_bad_label_duplicate_id_and_width(self):
        good = "001,B," + ",".join(map(str, range(30))) + "\n"
        for content in (good.replace(",B,", ",X,"), good + good,
                        good.replace(",29", ",nan"), "001,B,1,2\n"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "sample.data"
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_data(path)


if __name__ == "__main__":
    unittest.main()
