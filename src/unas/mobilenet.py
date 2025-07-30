import time
from typing import Final
from logging import Logger
from functools import cache
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.ops import Conv2dNormActivation

from .utils import make_caption

from .dep.torchprofile import profile_macs


__all__ = [
    "LAYER_SETTINGS",
    "FIRST_CONV_CHANNELS",
    "LAST_CONV_CHANNELS",
    "DROPOUT",
    "build_first",
    "build_last",
    "build_pool",
    "build_classififer",
    "skeletonnet_train",
    "skeletonnet_valid",
    "MobileNetV2Skeleton"
]


# MobileNetV2 settings for bottleneck layers. Copied from
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
LAYER_SETTINGS: Final[list[list[int]]] = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]

# The out channels of the first convolution layer of MobileNetV2.
FIRST_CONV_CHANNELS: Final[int] = 32

# The out channels of the last convolution layer of MobileNetV2.
LAST_CONV_CHANNELS: Final[int] = 1280


def build_first(
    out_channels: int = FIRST_CONV_CHANNELS,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> Conv2dNormActivation:
    """
    A convenience method to build the first conv2d layer of MobileNetV2.

    Args:
        out_channels (int):
            The amount of output channels.
        norm_layer (Callable[..., nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., nn.Module], None):
            The constructor for the activation layer.
    Returns:
        Conv2dNormActivation:
            The first conv2d layer.
    """
    return Conv2dNormActivation(
        3,
        out_channels,
        kernel_size=3,
        stride=2,
        norm_layer=norm_layer,
        activation_layer=activation_layer
    )


def build_last(
    in_channels: int = LAYER_SETTINGS[-1][1],
    out_channels: int = LAST_CONV_CHANNELS,
    norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm2d,
    activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
) -> Conv2dNormActivation:
    """
    A convenience method to build the last conv2d layer of MobileNetV2.

    Args:
        int_channels (int):
            The output channels of the last block operation.
        out_channels (int):
            The amount of output channels.
        norm_layer (Callable[..., nn.Module], None):
            The constructor for the norm layer.
        activation_layer (Callable[..., nn.Module], None):
            The constructor for the activation layer.
    Returns:
        Conv2dNormActivation:
            The last conv2d layer.
    """
    return Conv2dNormActivation(
        in_channels,
        out_channels,
        1,
        norm_layer=norm_layer,
        activation_layer=activation_layer
    )


def build_pool() -> nn.Sequential:
    """
    A convenience method to build the pooling layer of MobileNetV2.

    Returns:
        nn.Sequential:
            The combination of avg pooling and flatten operation.
    """
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1)
    )


class MobileSkeletonNet(nn.Module):
    """
    A skeleton of the MobileNetV2 architecture. Expects all layers to be passed
    to the constructor. Implements common methods for weight initialization,
    forward pass and computing FLOPs.

    The weight initialization is copied from
    https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
    """

    # Layers of the NN.
    first: nn.Module
    blocks: nn.ModuleList
    last: nn.Module
    pool: nn.Sequential  # avg pooling + flatten

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        initialize_weights: bool = True
    ) -> None:
        super().__init__()

        self.first = first
        self.blocks = nn.ModuleList(blocks)
        self.last = last
        self.pool = pool

        if initialize_weights:
            self._weight_initialization()

    def _weight_initialization(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.first(inputs)

        for block in self.blocks:
            inputs = block(inputs)

        inputs = self.last(inputs)

        return self.pool(inputs)

    @cache
    def flops(self, resolution: int, *, device=None) -> int:
        """
        The FLOPs/MACs of this model. They are calculated once and the result is
        cached for future uses.

        Args:
            resolution (int):
                The resolution of the input image.

        Returns:
            int:
                The computed FLOPs/MACs.
        """
        self.to(device)
        inputs = torch.randn(1, 3, resolution, resolution, device=device)

        return profile_macs(self, inputs)


def skeletonnet_train(
    skeletonnet: MobileSkeletonNet,
    dl: DataLoader,
    epochs: int,
    initial_lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 5e-5,
    *,
    logger: Logger,
    batches: int | None = None,
    device=None
) -> None:
    """
    Train a `MobileSkeletonNet` on the data set `ds` using the `SGD` optimizer.

    Args:
        skeletonnet (MobileSkeletonNet):
            The network to train.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set to train on.
        epochs (int):
            The number of training epochs.
        initial_lr (float):
            The learning rate intially set for `SGD`.
        momentum (float):
            The momenum for `SGD` to use.
        weight_decay (float):
            The weight decay for `SGD` to use.
        logger (Logger):
            The logger to output training status messages.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    train_start = time.time()

    batches = batches if batches is not None else len(dl)

    skeletonnet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        skeletonnet.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    skeletonnet.train()
    for i in range(epochs):
        logger.info(make_caption(f"Epoch {i + 1}/{epochs}", 70, "-"))
        epoch_start = time.time()

        batch_start = time.time()  # to capture data load time
        for j, (images, labels) in enumerate(dl):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = skeletonnet(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start

            logger.info(
                f"epoch={i + 1}, batch={j + 1:0{len(str(batches))}}/{batches}, time={batch_time:.2f}s"
            )
            batch_start = time.time()

            if batches == j + 1:
                break

            scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"time={epoch_time:.2f}s")

    train_time = time.time() - train_start
    logger.info(f"total={train_time:.2f}s")


def skeletonnet_valid(
    skeletonnet: MobileSkeletonNet,
    dl: DataLoader,
    *,
    logger: Logger,
    batches: int | None = None,
    device=None
) -> float:
    """
    Validate a `MobileSkeletonNet` on the data set `ds`.

    Args:
        skeletonnet (MobileSkeletonNet):
            The network to validate.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set used to validate.
        logger (Logger):
            The logger to output training status messages.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.

    Returns:
        float:
            The accuracy in [0, 1].
    """
    batches = batches if batches is not None else len(dl)

    skeletonnet.to(device)
    skeletonnet.eval()

    correct = 0
    total = 0
    valid_start = time.time()
    with torch.no_grad():
        for i, (features, labels) in enumerate(dl):
            features = features.to(device)
            labels = labels.to(device)

            outputs = skeletonnet(features)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if batches == i + 1:
                break

    logger.info(f"total={time.time() - valid_start:.2f}s")
    return float(correct) / float(total)
