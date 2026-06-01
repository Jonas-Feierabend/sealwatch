import pathlib
import conseal as cl 
import partial 
from conseal.lsb._costmap import Change

ASSETS_DIR = pathlib.Path('test/assets')
COVER_DIR = ASSETS_DIR / 'cover'
COVER_UNCOMPRESSED_GRAY_DIR = COVER_DIR / 'uncompressed_gray'
COVER_UNCOMPRESSED_COLOR_DIR = COVER_DIR / 'uncompressed_color'
COVER_COMPRESSED_GRAY_DIR = COVER_DIR / 'jpeg_75_gray'
COVER_COMPRESSED_COLOR_DIR = COVER_DIR / 'jpeg_75_color'

TEST_IMAGES = [
    'seal1',
    'seal2',
    'seal3',
    'seal4',
    'seal5',
    'seal6',
    'seal7',
    'seal8',
]


COST_FUNCTIONS = [
    ["hill",     cl.hill.simulate_single_channel],
    ["hugo",     cl.hugo.simulate_single_channel],
    ["lsbm",     partial(cl.lsb.simulate, modify=Change.LSB_MATCHING)],
    ["lsbr",     partial(cl.lsb.simulate, modify=Change.LSB_REPLACEMENT)],
    ["suniward", cl.suniward.simulate_single_channel],
    ["wow",      cl.wow.simulate_single_channel],
]