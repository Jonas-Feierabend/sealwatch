
import conseal as cl
import jpeglib
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sealwatch as sw
import sys
import tempfile
import torch
import unittest
sys.path.append('test')

import defs


class TestUcNet(unittest.TestCase):
    """Test suite for ucnet module."""
    _logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp

    # def test_structure(self):
    #     """Checks the shapes against the Tab. 2 in the paper."""
    #     self._logger.info('TestUcNet.test_structure()')
    #     model = sw.ucnet.UcNet()
    #     #
    #     x = torch.ones((1, 3, 256, 256), dtype=torch.float32)
    #     x_preprocessing = model.preprocessing(x)
    #     self.assertEqual(tuple(x_preprocessing.size()), (1, 186, 256, 256))
    #     #
    #     x_1a = model.block_1a(x_preprocessing)
    #     self.assertEqual(tuple(x_1a.size()), (1, 32, 256, 256))
    #     #
    #     x_2a = model.block_2a(x_1a)
    #     self.assertEqual(tuple(x_2a.size()), (1, 32, 128, 128))
    #     #
    #     x_3 = model.block_3(x_2a)
    #     self.assertEqual(tuple(x_3.size()), (1, 64, 64, 64))
    #     #
    #     x_2b = model.block_2b(x_3)
    #     self.assertEqual(tuple(x_2b.size()), (1, 128, 32, 32))
    #     #
    #     x_1b = model.block_1b(x_2b)
    #     self.assertEqual(tuple(x_1b.size()), (1, 256, 32, 32))
    #     #
    #     x_gap = model.gap(x_1b)
    #     self.assertEqual(tuple(x_gap.size()), (1, 256))
    #     x_fc = model.fc(x_gap)
    #     self.assertEqual(tuple(x_fc.size()), (1, 2))

    def test_pretrained(self):
        self._logger.info('TestUcNet.test_pretrained()')
        sw.ucnet.UcNet.pretrained()

    @parameterized.expand([[fname] for fname in defs.TEST_IMAGES])
    def test_infere_xunet(self, fname: str):
        self._logger.info(f'TestUcNet.test_infere_xunet({fname=})')
        #
        DEVICE = torch.device('cpu')
        model = sw.ucnet.UcNet.pretrained().to(DEVICE)

        #
        cover_path = defs.COVER_UNCOMPRESSED_COLOR_DIR / f'{fname}.png'
        x0 = np.array(Image.open(cover_path))
        s0 = sw.ucnet.infere_single(x0, model=model, device=DEVICE)

        # load cover
        x1 = cl.lsb.simulate(x0, alpha=1., modify=cl.LSB_MATCHING, seed=12345)
        s1 = sw.ucnet.infere_single(x1, model=model, device=DEVICE)
        # self.assertLess(y0, y1)
        print(fname, s0, s1)
