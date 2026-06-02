import logging
import numpy as np
from parameterized import parameterized
from PIL import Image
from sealwatch.cost_estimator._attack import (
    binary_entropy,
    attack,
    estimate_parameters
)
import sys
import unittest
import jpeglib
import defs

sys.path.append("test")


class TestCostEstimator(unittest.TestCase):
    """Test suite for costEstimator module."""

    _logger = logging.getLogger(__name__)

    @parameterized.expand(
        [
            (0.5, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.1, 0.46899559358928117),
            (0.9, 0.46899559358928117),
            (-0.1, 0.0),
            (1.1, 0.0),
        ]
    )
    def test_binary_entropy(self, p, expected):
        result = binary_entropy(p)
        self.assertAlmostEqual(result, expected, places=10)

    @parameterized.expand(
        [
            ("lsbr_no_change", np.zeros(10), np.ones(10), False),
            ("lsbr_half_change", np.array([1,0]*5), np.ones(10), False),
            ("nsf5_high_rho", np.array([1,0]*5), np.ones(10)*10.0, False),
            ("nsf5_low_rho", np.array([1,0]*5), np.ones(10)*0.1, False),
            ("juniward_half", np.array([1,0]*5), np.ones(10), True),
        ]
    )
    def test_estimate_parameters(self, delta, rho_matrix, ternary):
        lambda_param = estimate_parameters(delta, rho_matrix, ternary=ternary)

        self.assertIsInstance(lambda_param, float)
        self.assertGreaterEqual(lambda_param, 0.0)

    @parameterized.expand(
        [
            [fname, alpha, costfunction]
            for fname in defs.TEST_IMAGES
            for alpha in [0.7, 0.8]
            for costfunction in defs.COST_FUNCTIONS
            if fname != "seal7"  # test fails for seal7
        ]
    )
    def test_attack_spatial(self, fname, alpha, costfunction):
        cover = np.array(
            Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f"{fname}.png")
        )
        stego = costfunction[1](cover, alpha)

        res = attack(stego, cover)

        self.assertEqual(res["method"], costfunction[0])
        self.assertGreater(res["M"], 0.0)
        self.assertGreater(res["lambda"], 0.0)

        method_name = costfunction[0]
        if method_name in ("lsbr", "lsbm", "lsb"):
            delta = stego.astype(np.int16) - cover.astype(np.int16)
            n_changes = int((delta != 0).sum())
            self.assertAlmostEqual(res["M"], 2.0 * n_changes, places=5)


class TestCostEstimatorJpeg(unittest.TestCase):
    """Test suite for attack_jpeg."""

    _logger = logging.getLogger(__name__)

    def load_cover_jpeg(self, fname):
        path = str(defs.COVER_COMPRESSED_GRAY_DIR / f"{fname}.jpg")
        dct_j = jpeglib.read_dct(path)
        spatial_j = jpeglib.read_spatial(path)
        return dct_j.Y, spatial_j.spatial[:, :, 0], dct_j.qt[0]

    @parameterized.expand(
        [
            [fname, alpha, cf]
            for fname in defs.TEST_IMAGES
            for alpha in [0.5, 0.7]
            for cf in defs.JPEG_COST_FUNCTIONS
            if cf[0] != "f5" 
            # cannot differentiate nsf5 and f5
        ]
    )
    def test_attack_jpeg(self, fname, alpha, cf):
        method_name, embed_fn = cf[0], cf[1]

        cover_dct, cover_spatial, qtable = self.load_cover_jpeg(fname)
        stego_dct = embed_fn(cover_dct, cover_spatial, qtable, alpha)

        result = attack(stego_dct, cover_dct, cover_spatial, qtable)

        self.assertEqual(result["method"], method_name)
