
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import time
import unittest

sys.path.append('test')
import defs


class TestCostEstimator(unittest.TestCase):
    """Test suite for hcfcom module."""
    _logger = logging.getLogger(__name__)



    @parameterized.expand([[f] for f in defs.TEST_IMAGES])
    def test_extract(self, fname):
        self._logger.info(f'TestCostEstimator')

    
        x0 = np.array(Image.open(defs.COVER_UNCOMPRESSED_GRAY_DIR / f'dolphin.png'))
        ret = sw.cost_estimator.attack(x0, x0)

 
        self.assertEqual(1,2)

