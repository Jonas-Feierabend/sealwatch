import pathlib
import conseal as cl
from functools import partial
from conseal.lsb._costmap import Change

ASSETS_DIR = pathlib.Path("test/assets")
COVER_DIR = ASSETS_DIR / "cover"
COVER_UNCOMPRESSED_GRAY_DIR = COVER_DIR / "uncompressed_gray"
COVER_UNCOMPRESSED_COLOR_DIR = COVER_DIR / "uncompressed_color"
COVER_COMPRESSED_GRAY_DIR = COVER_DIR / "jpeg_75_gray"
COVER_COMPRESSED_COLOR_DIR = COVER_DIR / "jpeg_75_color"

TEST_IMAGES = [
    "seal1",
    "seal2",
    "seal3",
    "seal4",
    "seal5",
    "seal6",
    "seal7",
    "seal8",
]


COST_FUNCTIONS = [
    ["hill", cl.hill.simulate_single_channel],
    ["hugo", cl.hugo.simulate_single_channel],
    ["lsbm", partial(cl.lsb.simulate, modify=Change.LSB_MATCHING)],
    ["lsbr", partial(cl.lsb.simulate, modify=Change.LSB_REPLACEMENT)],
    ["suniward", cl.suniward.simulate_single_channel],
    ["wow", cl.wow.simulate_single_channel],
]

JPEG_COST_FUNCTIONS = [
    [
        "juniward",
        lambda dct, spatial, qt, alpha: cl.juniward.simulate_single_channel(
            x0=spatial, y0=dct, qt=qt, alpha=alpha
        ),
    ],
    [
        "uerd",
        lambda dct, spatial, qt, alpha: cl.uerd.simulate_single_channel(
            y0=dct, qt=qt, alpha=alpha
        ),
    ],
    [
        "ebs",
        lambda dct, spatial, qt, alpha: cl.ebs.simulate_single_channel(
            y0=dct, qt=qt, alpha=alpha
        ),
    ],
    [
        "nsf5",
        lambda dct, spatial, qt, alpha: cl.nsF5.simulate_single_channel(
            y0=dct, alpha=alpha
        ),
    ],
    [
        "f5",
        lambda dct, spatial, qt, alpha: cl.F5.simulate_single_channel(
            y0=dct, alpha=alpha
        ),
    ],
    [
        "lsb",
        lambda dct, spatial, qt, alpha: cl.lsb.simulate(
            dct, modify=Change.LSB_REPLACEMENT, alpha=alpha
        ),
    ],
]
