import logging
import numpy as np
from parameterized import parameterized
from PIL import Image
from sealwatch.cost_estimator._attack import (
    attack,
    estimate_lambda,
    estimate_lambda_directional,
    _log_likelihood,
    _log_likelihood_directional,
    _is_impossible,
)
import sys
import unittest
import jpeglib
import defs

sys.path.append("test")


class TestEstimateParameters(unittest.TestCase):
    """Unit tests for estimate_lambda."""

    _logger = logging.getLogger(__name__)

    @parameterized.expand(
        [
            ("binary_no_change", np.zeros(10), np.ones(10), False),
            ("binary_half_change", np.array([1, 0] * 5), np.ones(10), False),
            (
                "binary_high_rho",
                np.array([1, 0] * 5),
                np.ones(10) * 10.0,
                False,
            ),
            ("binary_low_rho", np.array([1, 0] * 5), np.ones(10) * 0.1, False),
            ("ternary_half", np.array([1, 0] * 5), np.ones(10), True),
            (
                "ternary_high_rho",
                np.array([1, 0] * 5),
                np.ones(10) * 10.0,
                True,
            ),
        ]
    )
    def test_returns_nonneg_float(self, name, delta, rho_matrix, ternary):
        """Lambda must be a non-negative float."""
        lam = estimate_lambda(delta, rho_matrix, ternary=ternary)
        self.assertIsInstance(lam, float)
        self.assertGreaterEqual(lam, 0.0)

    @parameterized.expand(
        [
            ("binary", np.array([1, 0] * 5), np.ones(10), False),
            ("ternary", np.array([1, 0] * 5), np.ones(10), True),
            (
                "binary_varied",
                np.array([1, 0] * 5),
                np.array([0.5, 2.0] * 5),
                False,
            ),
        ]
    )
    def test_beta_theo_matches_beta_obs(
        self, name, delta, rho_matrix, ternary
    ):
        """After optimization beta_theo must match beta_obs within tolerance."""
        lam = estimate_lambda(delta, rho_matrix, ternary=ternary, max_iter=200)
        exponent = np.exp(-lam * rho_matrix)
        if ternary:
            p_i = exponent / (1 + 2 * exponent)
            beta_theo = float(np.mean(2 * p_i))
        else:
            p_i = exponent / (1 + exponent)
            beta_theo = float(np.mean(p_i))
        beta_obs = float((delta != 0).mean())
        self.assertAlmostEqual(beta_theo, beta_obs, places=2)

    def test_no_changes_returns_zero(self):
        """Zero observed changes must return lambda=0."""
        lam = estimate_lambda(np.zeros(10), np.ones(10))
        self.assertEqual(lam, 0.0)


class TestEstimateParametersDirectional(unittest.TestCase):
    """Unit tests for estimate_lambda_directional."""

    @parameterized.expand(
        [
            (
                "symmetric",
                np.array([1, 0, -1, 0] * 3),
                np.ones(12),
                np.ones(12),
            ),
            (
                "asymmetric",
                np.array([1, 0, -1, 0] * 3),
                np.ones(12) * 0.5,
                np.ones(12) * 2.0,
            ),
            (
                "high_rho",
                np.array([1, 0] * 6),
                np.ones(12) * 5.0,
                np.ones(12) * 5.0,
            ),
        ]
    )
    def test_returns_nonneg_float(self, name, delta, rho_p1, rho_m1):
        lam = estimate_lambda_directional(delta, rho_p1, rho_m1)
        self.assertIsInstance(lam, float)
        self.assertGreaterEqual(lam, 0.0)

    def test_beta_theo_matches_beta_obs(self):
        """After optimization beta_theo must match beta_obs within tolerance."""
        delta = np.array([1, 0, -1, 0] * 5, dtype=float)
        rho_p1 = np.ones(20) * 0.5
        rho_m1 = np.ones(20) * 2.0
        lam = estimate_lambda_directional(delta, rho_p1, rho_m1)
        exp_p1 = np.exp(-lam * rho_p1)
        exp_m1 = np.exp(-lam * rho_m1)
        denom = 1 + exp_p1 + exp_m1
        beta_theo = float(np.mean((exp_p1 + exp_m1) / denom))
        beta_obs = float((delta != 0).mean())
        self.assertAlmostEqual(beta_theo, beta_obs, places=4)

    def test_no_changes_returns_zero(self):
        lam = estimate_lambda_directional(
            np.zeros(10), np.ones(10), np.ones(10)
        )
        self.assertEqual(lam, 0.0)


class TestLogLikelihood(unittest.TestCase):
    """Unit tests for _log_likelihood."""

    def test_returns_float(self):
        delta = np.array([1, 0, -1, 0])
        exp = np.full(4, 0.5)
        result = _log_likelihood(delta, exp, ternary=True)
        self.assertIsInstance(result, float)

    def test_all_unchanged_higher_than_all_changed(self):
        """When nothing changed, unchanged model should score higher."""
        delta_none = np.zeros(10)
        delta_all = np.ones(10)
        exp = np.full(10, 0.1)  # low change probability
        llh_none = _log_likelihood(delta_none, exp, ternary=True)
        llh_all = _log_likelihood(delta_all, exp, ternary=True)
        self.assertGreater(llh_none, llh_all)

    def test_binary_vs_ternary(self):
        """Binary and ternary models must both return finite floats."""
        delta = np.array([1, 0, 1, 0])
        exp = np.full(4, 0.3)
        llh_bin = _log_likelihood(delta, exp, ternary=False)
        llh_ter = _log_likelihood(delta, exp, ternary=True)
        self.assertTrue(np.isfinite(llh_bin))
        self.assertTrue(np.isfinite(llh_ter))


class TestLogLikelihoodDirectional(unittest.TestCase):
    """Unit tests for _log_likelihood_directional."""

    def test_returns_float(self):
        delta = np.array([1, 0, -1, 0])
        exp_p1 = np.full(4, 0.4)
        exp_m1 = np.full(4, 0.2)
        result = _log_likelihood_directional(delta, exp_p1, exp_m1)
        self.assertIsInstance(result, float)

    def test_direction_matters(self):
        """Higher exp for the observed direction -> higher LLH."""
        delta = np.array([1, 1, 1, 1])  # all +1
        exp_high_p1 = np.full(4, 0.9)
        exp_low_p1 = np.full(4, 0.1)
        exp_m1 = np.full(4, 0.1)
        llh_high = _log_likelihood_directional(delta, exp_high_p1, exp_m1)
        llh_low = _log_likelihood_directional(delta, exp_low_p1, exp_m1)
        self.assertGreater(llh_high, llh_low)


class TestIsImpossible(unittest.TestCase):
    """Unit tests for _is_impossible."""

    def _make_dct(self, values):
        arr = np.zeros((2, 2, 8, 8), dtype=np.int32)
        arr.ravel()[: len(values)] = values
        return arr

    def test_lsb_even_decreases_impossible(self):
        """LSB-R: even cover coefficient decreased by 1 is impossible."""
        cover = self._make_dct([2])  # even
        delta = self._make_dct([-1])  # decrease
        self.assertTrue(_is_impossible("lsb", cover, delta, False, False))

    def test_lsb_odd_increases_impossible(self):
        """LSB-R: odd cover coefficient increased by 1 is impossible."""
        cover = self._make_dct([3])  # odd
        delta = self._make_dct([+1])  # increase
        self.assertTrue(_is_impossible("lsb", cover, delta, False, False))

    def test_lsb_valid_not_impossible(self):
        cover = self._make_dct([2])  # even
        delta = self._make_dct([+1])  # increase — valid for even
        self.assertFalse(_is_impossible("lsb", cover, delta, False, False))

    def test_nsf5_wrong_direction_impossible(self):
        cover = self._make_dct([0])
        delta = self._make_dct([0])
        self.assertTrue(_is_impossible("nsf5", cover, delta, True, False))

    def test_f5_zero_changed_impossible(self):
        cover = self._make_dct([0])
        delta = self._make_dct([0])
        self.assertTrue(_is_impossible("f5", cover, delta, False, True))

    def test_f5_wrong_direction_also_impossible(self):
        cover = self._make_dct([0])
        delta = self._make_dct([0])
        self.assertTrue(_is_impossible("f5", cover, delta, True, False))

    def test_juniward_never_impossible(self):
        cover = self._make_dct([0])
        delta = self._make_dct([0])
        self.assertFalse(_is_impossible("juniward", cover, delta, True, True))


class TestAttackSpatial(unittest.TestCase):
    """Integration tests for attack (spatial domain)."""

    _logger = logging.getLogger(__name__)

    @parameterized.expand(
        [
            [fname, alpha, costfunction]
            for fname in defs.TEST_IMAGES
            for alpha in [0.3, 0.5, 0.7]
            for costfunction in defs.COST_FUNCTIONS
            if fname != "seal7"
            or alpha <= 0.5  # seal7 unreliable at high alpha
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

        if costfunction[0] == "lsbr":
            delta = stego.astype(np.int16) - cover.astype(np.int16)
            n_changes = int((delta != 0).sum())
            self.assertAlmostEqual(res["M"], 2.0 * n_changes, places=5)

    def test_identical_images_raises(self):
        cover = np.zeros((64, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            attack(cover, cover)


class TestAttackJpeg(unittest.TestCase):
    """Integration tests for attack (JPEG domain)."""

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
            for alpha in [0.3, 0.5, 0.7]
            for cf in defs.JPEG_COST_FUNCTIONS
            if cf[0] != "f5"
            # f5 and nsf5 are statistically indistinguishable:
            # both use uniform-cost unidirectional embedding.
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

        if method_name == "lsb":
            embed_mask = cover_dct != 0
            embed_mask[:, :, 0, 0] = False
            delta = stego_dct.astype(np.int32) - cover_dct.astype(np.int32)
            n_changes = int((delta[embed_mask] != 0).sum())
            self.assertAlmostEqual(result["M"], 2.0 * n_changes, places=5)

    def test_identical_images_raises(self):
        cover = np.zeros((4, 4, 8, 8), dtype=np.int32)
        with self.assertRaises(ValueError):
            attack(cover, cover, np.zeros((64, 64)), np.ones((8, 8)))


class TestAttackDispatch(unittest.TestCase):
    """Smoke tests for the auto-dispatching attack() function."""

    def test_spatial_dispatch(self):
        """2D input must route to spatial attack and return expected keys."""
        cover = np.array(
            Image.open(
                defs.COVER_UNCOMPRESSED_GRAY_DIR / f"{defs.TEST_IMAGES[0]}.png"
            )
        )
        stego = defs.COST_FUNCTIONS[0][1](cover, 0.5)
        res = attack(stego, cover)
        self.assertIn("method", res)
        self.assertIn("M", res)
        self.assertIn("lambda", res)
        self.assertIn("all", res)

    def test_jpeg_dispatch(self):
        """4D input must route to JPEG attack and return expected keys."""
        fname = defs.TEST_IMAGES[0]
        path = str(defs.COVER_COMPRESSED_GRAY_DIR / f"{fname}.jpg")
        dct_j = jpeglib.read_dct(path)
        spatial_j = jpeglib.read_spatial(path)
        cover_dct = dct_j.Y
        cover_spatial = spatial_j.spatial[:, :, 0]
        qtable = dct_j.qt[0]
        cf = defs.JPEG_COST_FUNCTIONS[0]
        stego_dct = cf[1](cover_dct, cover_spatial, qtable, 0.5)
        res = attack(stego_dct, cover_dct, cover_spatial, qtable)
        self.assertIn("method", res)
        self.assertIn("M", res)
        self.assertIn("lambda", res)
        self.assertIn("all", res)
