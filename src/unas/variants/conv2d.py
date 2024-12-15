from collections.abc import Callable

import torch.nn as nn

import numpy as np

from ..searchspace import (
    KERNEL_CHOICES,
    SE_CHOICES,
    SHORTCUT_CHOICES,
    LAYER_CHOICES,
    CHANNEL_CHOICES,
    EXPANSION_CHOICES,
    uniform_layers,
    uniform_model,
    ConvOp,
    Block,
    Model,
    SearchSpace
)
from ..oneshot import ChoiceBlock
from ..oneshot.layers import SharedConv2dModule
from ..mobilenet import FIRST_CONV_CHANNELS, LAYER_SETTINGS
from ..utils import make_divisible


__all__ = ["conv2d_choice_block_ctor", "Conv2dSearchSpace"]


def conv2d_choice_block_ctor(
    in_channels: int,
    out_channels: int,
    width_mult: float,
    round_nearest: int,
    stride: int,
    activation_layer: Callable[..., nn.Module] | None
) -> tuple[int, ChoiceBlock]:
    out_channels = make_divisible(
        out_channels * max(CHANNEL_CHOICES) * width_mult, round_nearest
    )

    return (
        out_channels,
        ChoiceBlock(
            in_channels,
            out_channels,
            1,
            max(SE_CHOICES),
            max(KERNEL_CHOICES),
            max(LAYER_CHOICES),
            stride,
            activation_layer,
            SharedConv2dModule
        )
    )


def _sample_block(
    rng: np.random.Generator,
    depth: int,
    in_channels: int,
    out_channels: int,
    first_stride: int
) -> Block:
    """
    Randomly amples a single instance of a `Block` from all choices
    possible, except the used convolution operation. Keep that fixed
    at `ConvOp.CONV2D`.

    Args:
        rng (numpy.random.Generator):
            A random number generator to uniformly distribute the blocks
            parameters.
        depth (int):
            The number of layers.
        in_channels (int):
            The number of in channels.
        out_channels (int):
            The number of out channels.
        first_stride (int):
            The stride for the first layer.

    Returns:
        Block:
            The newly created block.
    """
    op = ConvOp.CONV2D

    choice = rng.integers(0, len(KERNEL_CHOICES))
    kernel_size = KERNEL_CHOICES[choice]

    choice = rng.integers(0, len(SE_CHOICES))
    se_ratio = SE_CHOICES[choice]

    choice = rng.integers(0, len(SHORTCUT_CHOICES))
    shortcut = SHORTCUT_CHOICES[choice]

    expansion_ratio = 1

    layers = uniform_layers(
        op,
        in_channels,
        out_channels,
        kernel_size,
        first_stride,
        expansion_ratio,
        shortcut,
        se_ratio,
        depth
    )

    return Block(layers)


def _sample_model(rng: np.random.Generator, width_mult: float) -> Model:
    """
    Randomly samples a single instance of a `Model` afrom the `ConvOp.Conv2d`
    search space. The backbone of the search space is MobileNetV2, so use its
    layer settings and input and output channels.

    Args:
        rng (numpy.random.Generator):
            A random number generator to uniformly distribute the models
            parameters.
        width_mult (float):
            Modifies the input and output channels of the hidden layers.

    Returns:
        Model:
            The newly created model.
    """
    blocks: list[Block] = []

    round_nearest = 8

    # As per MobileNetV2 architecture. Copied from
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    in_channels = make_divisible(
        FIRST_CONV_CHANNELS * width_mult, round_nearest
    )

    for settings in LAYER_SETTINGS:
        depth = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

        choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
        out_channels = make_divisible(
            settings[1] * choice * width_mult, round_nearest
        )

        blocks.append(
            _sample_block(rng, depth, in_channels, out_channels, settings[3])
        )

        in_channels = out_channels

    return Model(width_mult, blocks)


def mutator(i: int, width_mult: float, block: Block) -> Block:
    rng = np.random.default_rng()

    depth = LAYER_CHOICES[rng.integers(0, len(LAYER_CHOICES))]

    choice = CHANNEL_CHOICES[rng.integers(0, len(CHANNEL_CHOICES))]
    out_channels = make_divisible(
        LAYER_SETTINGS[i][1] * choice * width_mult, 8
    )

    new = _sample_block(
        rng,
        depth,
        block.layers[0].in_channels,
        out_channels,
        LAYER_SETTINGS[i][3]
    )

    return new


class Conv2dSearchSpace(SearchSpace):
    def sample_model(self, rng: np.random.Generator) -> Model:
        return _sample_model(rng, self.width_mult)

    def max_models(self) -> tuple[Model, ...]:
        return (
            uniform_model(
                ConvOp.CONV2D,
                max(KERNEL_CHOICES),
                max(SE_CHOICES),
                False,
                max(LAYER_CHOICES),
                max(CHANNEL_CHOICES),
                max(EXPANSION_CHOICES),
                self.width_mult
            ),
        )
