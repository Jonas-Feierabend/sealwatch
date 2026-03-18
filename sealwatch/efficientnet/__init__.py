"""

Author: Martin Benes
"""

import numpy as np
import torch

#
from . import _b0
from . import _b4
#
from ._b0 import B0
from ._b4 import B4


def infere_single(
    x: np.ndarray,
    model: torch.nn.Module = None,
    *,
    device: torch.device = torch.device('cpu'),
) -> np.ndarray:
    """Runs inference for a single image.

    :param x: image
    :type x:
    :param model:
    :type model:
    :param device:
    :type device:
    :return:
    :rtype:
    """
    # prepare data
    x = (x / 255.).transpose(2, 0, 1)[None]
    x_ = torch.from_numpy(x).to(dtype=torch.float32, device=device)

    #
    assert not model.training, 'Model must be in eval mode for inference.'
    with torch.no_grad():
        # infere
        logit = model(x_)
        y_ = torch.nn.functional.softmax(logit, dim=1)[0, 1]
        # convert back to numpy
        y = y_.detach().cpu().numpy()
    #
    return y


__all__ = [
    'B0',
    # 'pretrained',
    # 'infere_single',
]