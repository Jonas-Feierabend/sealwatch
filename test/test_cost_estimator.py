
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
from sealwatch.cost_estimator._attack import binary_entropy, attack, attack_jpeg
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs


class TestCostEstimator(unittest.TestCase):
    """Test suite for costEstimator module."""
    _logger = logging.getLogger(__name__)




    @parameterized.expand([
        (0.5,  1.0),
        (0.0,  0.0),
        (1.0,  0.0),
        (0.1,  0.46899559358928117),
        (0.9,  0.46899559358928117),  
        (-0.1, 0.0),
        (1.1,  0.0),
    ])
    def test_binary_entropy(self, p, expected):
        result = binary_entropy(p)
        self.assertAlmostEqual(result, expected, places=10)


    @parameterized.expand([
        [fname, alpha, costfunction]
        for fname in defs.TEST_IMAGES
        for alpha in [.7, .8]
        for costfunction in defs.COST_FUNCTIONS 
    ])
    def test_attack_spatial(self,fname,alpha, costfunction): 
        cover = np.array(Image.open(fname).convert('L'))
        stego = costfunction[1](cover, alpha)

        res = attack(stego, cover)

        self.assertEqual(res["method"],costfunction[0]) 

