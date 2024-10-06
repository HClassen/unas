from collections.abc import Callable

import torch.nn as nn

from ..mobilenet import (
    build_first,
    build_last,
    build_pool,
    MobileSkeletonNet
)

from . import Model
from .layers import layer_to_module


__all__ = ["build_model"]


def build_model(
    model: Model,
    classes: int,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
    initialize_weights: bool = True
) -> MobileSkeletonNet:
    """
    Build a `MobileSkeletonNet` from a given `Model`.

    Args:
        model (Model):
            A model sampled from the MnasNet search space.
        classes (int).
            The amout of classes for the classifier to recognize.
        dropout (float):
            The percentage of dropout used in the classifier.
        norm_layer (Callable[..., torch.nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., torch.nn.Module], None):
            The constructor for the activation layer.

    Returns:
        MobileSkeletonNet:
            The created Pytorch module.
    """
    in_channels = model.blocks[0].layers[0].in_channels
    first = build_first(in_channels, norm_layer, activation_layer)

    blocks: list[nn.Sequential] = []
    for block in model.blocks:
        blocks.append(
            nn.Sequential(
                *[
                    layer_to_module(layer, norm_layer, activation_layer)
                    for layer in block.layers
                ]
            )
        )

    in_channels = model.blocks[-1].layers[-1].out_channels
    last = build_last(
        in_channels, classes, norm_layer, activation_layer
    )

    pool = build_pool()

    return MobileSkeletonNet(first, blocks, last, pool, initialize_weights)
