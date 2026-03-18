"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import numpy as np
import timm
import torch


class B0(torch.nn.Module):
    """"""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        no_stem_stride: bool = True,
        weight_seed: int = None,
        **kw,
    ):
        super().__init__()
        with torch.random.fork_rng():
            if weight_seed is not None:
                torch.manual_seed(weight_seed)
            backbone = timm.create_model(
                'efficientnet_b0',
                in_chans=in_channels,
                num_classes=out_channels,
                **kw,
            )
        if no_stem_stride:
            backbone.conv_stem.stride = (1, 1)
        self.add_module('backbone', backbone)
        self.default_cfg = backbone.default_cfg

    def state_dict(self, *args, **kw):
        out = super().state_dict(*args, **kw)
        return {k.replace('backbone.', ''): v for k, v in out.items()}

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = {f'backbone.{k}': v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, x):
        """Must be defined explicitly."""
        return self.backbone(x)

    def __repr__(self) -> str:
        return repr(self.backbone)

    @classmethod
    def pretrained(cls, in_channels: int = 1, out_channels: int = 2, **kw):
        """"""
        # create model
        model = cls(in_channels=in_channels, out_channels=out_channels, **kw)

        # download weights
        b0_url = model.default_cfg['url']
        state_dict = torch.hub.load_state_dict_from_url(b0_url, map_location='cpu')

        # adapt input channels
        if state_dict['conv_stem.weight'].size(1) != in_channels:
            state_dict['conv_stem.weight'] = timm.models.adapt_input_conv(
                in_channels, state_dict['conv_stem.weight'])

        # remove FC
        model_state_dict = model.state_dict()
        state_dict['classifier.weight'] = model_state_dict['classifier.weight']
        state_dict['classifier.bias'] = model_state_dict['classifier.bias']

        # load weights
        model.load_state_dict(state_dict, strict=True)
        logging.info(f'using pretrained weights from {b0_url}')
        return model


def pretrained(*args, **kw) -> B0:
    return B0.pretrained(*args, **kw)

