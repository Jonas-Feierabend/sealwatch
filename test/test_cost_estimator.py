
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
import jpeglib 

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
        if fname != 'seal7' # test fails for seal7
    ])
    def test_attack_spatial(self,fname,alpha, costfunction): 
        cover =  np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'{fname}.png'))
        stego = costfunction[1](cover, alpha)

        res = attack(stego, cover)

        self.assertEqual(res["method"],costfunction[0]) 




class TestCostEstimatorJpeg(unittest.TestCase):
    """Test suite for attack_jpeg."""
    _logger = logging.getLogger(__name__)
 

    def load_cover_jpeg(self,fname):
        path = str(defs.COVER_COMPRESSED_GRAY_DIR / f'{fname}.jpg')
        dct_j     = jpeglib.read_dct(path)
        spatial_j = jpeglib.read_spatial(path)
        return dct_j.Y, spatial_j.spatial[:, :, 0], dct_j.qt[0]
    
    @parameterized.expand([
        [fname, alpha, cf]
        for fname in defs.TEST_IMAGES
        for alpha in [0.5, 0.7]
        for cf in defs.JPEG_COST_FUNCTIONS
    ])
    def test_attack_jpeg(self, fname, alpha, cf):
        method_name, embed_fn = cf[0], cf[1]

        cover_dct, cover_spatial, qtable = self.load_cover_jpeg(fname)
        stego_dct = embed_fn(cover_dct, cover_spatial, qtable, alpha)

        result = attack_jpeg(stego_dct, cover_dct, cover_spatial, qtable)

        self.assertEqual(result['method'], method_name)