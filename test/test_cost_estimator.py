import logging
import numpy as np
from parameterized import parameterized
from PIL import Image
from sealwatch.cost_estimator._attack import (
    attack,
    estimate_parameters
)
import sys
import unittest
import jpeglib
import defs

sys.path.append("test")


class TestCostEstimatorUtilities(unittest.TestCase): 
    """Test site for costEstimator module."""


    _logger = logging.getLogger(__name__)

    @parameterized.expand(
        [
            ("symmetric", np.array([1, 0, -1]*5), np.ones(15), np.ones(15)),
            ("asymmetric_high_p1", np.array([1, 0, -1]*5), np.ones(15)*10.0, np.ones(15)*0.1),
            ("asymmetric_high_m1", np.array([1, 0, -1]*5), np.ones(15)*0.1, np.ones(15)*10.0),
            ("no_change", np.zeros(15), np.ones(15), np.ones(15)),
        ]
    )
    def test_estimate_parameters_directional(self, name, delta, rho_p1, rho_m1):
        from sealwatch.cost_estimator._attack import estimate_parameters_directional
        lambda_param = estimate_parameters_directional(delta, rho_p1, rho_m1)

        self.assertIsInstance(lambda_param, float)
        self.assertGreaterEqual(lambda_param, 0.0)

    @parameterized.expand(
        [
            ("all_plus", np.array([1, 0, 1, 0]), np.ones(4), np.ones(4)),
            ("all_minus", np.array([-1, 0, -1, 0]), np.ones(4), np.ones(4)),
            ("mixed", np.array([1, -1, 0, 1]), np.ones(4), np.ones(4)),
            ("no_change", np.zeros(4), np.ones(4), np.ones(4)),
        ]
    )
    def test_log_likelihood_directional(self, name, delta, exp_p1, exp_m1):
        from sealwatch.cost_estimator._attack import _log_likelihood_directional
        log_lik = _log_likelihood_directional(delta, exp_p1, exp_m1)

        self.assertIsInstance(log_lik, float)
        self.assertLessEqual(log_lik, 0.0) 

    def test_log_likelihood_directional_higher_for_correct_direction(self):
        """Correct direction should yield higher log-likelihood."""
        from sealwatch.cost_estimator._attack import _log_likelihood_directional
        delta = np.array([+1, 0, +1, 0])

        exp_p1_good = np.ones(4) * 10.0
        exp_m1_good = np.ones(4) * 0.1


        exp_p1_bad = np.ones(4) * 0.1
        exp_m1_bad = np.ones(4) * 10.0

        lik_good = _log_likelihood_directional(delta, exp_p1_good, exp_m1_good)
        lik_bad  = _log_likelihood_directional(delta, exp_p1_bad,  exp_m1_bad)

        self.assertGreater(lik_good, lik_bad)

    @parameterized.expand(
        [
            ("lsbr_no_change", np.zeros(10), np.ones(10), False),
            ("lsbr_half_change", np.array([1,0]*5), np.ones(10), False),
            ("nsf5_high_rho", np.array([1,0]*5), np.ones(10)*10.0, False),
            ("nsf5_low_rho", np.array([1,0]*5), np.ones(10)*0.1, False),
            ("juniward_half", np.array([1,0]*5), np.ones(10), True),
        ]
    )
    def test_estimate_parameters(self, name, delta, rho_matrix, ternary):
        lambda_param = estimate_parameters(delta, rho_matrix, ternary=ternary)

        self.assertIsInstance(lambda_param, float)
        self.assertGreaterEqual(lambda_param, 0.0)

class TestCostEstimatorSpatial(unittest.TestCase):
    """Test suite for costEstimator module."""

    _logger = logging.getLogger(__name__)



    @parameterized.expand(
        [
            [fname, alpha, costfunction]
            for fname in defs.TEST_IMAGES
            for alpha in [0.7, 0.8]
            for costfunction in defs.COST_FUNCTIONS
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
        if method_name == "lsbr":
            delta = stego.astype(np.int16) - cover.astype(np.int16)
            n_changes = int((delta != 0).sum())
            self.assertAlmostEqual(res["M"], 2.0 * n_changes, places=5)


class TestCostEstimatorDCT(unittest.TestCase):
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
            # because of the implementation of f5/nsf5. 
            # f5 is nsf5 with a slightly higher embedding rate. 
            # because of that it cannot be differentiated 
        ]
    )
    def test_attack_jpeg(self, fname, alpha, cf):
            method_name, embed_fn = cf[0], cf[1]

            cover_dct, cover_spatial, qtable = self.load_cover_jpeg(fname)
            stego_dct = embed_fn(cover_dct, cover_spatial, qtable, alpha)

            result = attack(stego_dct, cover_dct, cover_spatial, qtable)

            self.assertEqual(result["method"], method_name)
            self.assertGreater(result["M"], 0.0)
            self.assertGreater(result["lambda"], 0.0)

            if method_name in ("lsb",):
                embed_mask = cover_dct != 0
                embed_mask[:, :, 0, 0] = False
                delta = stego_dct.astype(np.int32) - cover_dct.astype(np.int32)
                n_changes = int((delta[embed_mask] != 0).sum())
                self.assertAlmostEqual(result["M"], 2.0 * n_changes, places=5)
