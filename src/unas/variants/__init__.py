from collections.abc import Callable

import torch
import torch.nn as nn

from ..searchspace import LAYER_CHOICES, ConvOp, Model
from ..searchspace.layers import LayerName, BaseModule
from ..oneshot import WarmUpCtx, SuperNet
from ..oneshot.helper import (
    l1_norm,
    l1_reorder_conv2d,
    l1_reorder_batchnorm2d,
    l1_reorder_se,
    l1_reorder_main,
    has_norm
)
from ..mobilenet import MobileSkeletonNet


def _to_template(
    conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d
) -> tuple[int, int, tuple[int, int], bool, bool]:
    if conv2d.out_channels != bn2d.num_features:
        raise RuntimeError(
            "out channels and num features missmatch: "
            f"{conv2d.out_channels} != {bn2d.num_features}"
        )

    return (
        conv2d.in_channels,
        conv2d.out_channels,
        conv2d.kernel_size,
        conv2d.bias is not None,
        bn2d.track_running_stats
    )


def _from_template(
    template: tuple[int, int, tuple[int, int], bool, bool], device
) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
    in_channels, out_channels, kernel_size, bias, track_running_stats = \
        template

    conv2d = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        bias=bias,
        device=device
    )

    bn2d = nn.BatchNorm2d(
        out_channels,
        track_running_stats=track_running_stats,
        device=device
    )

    return conv2d, bn2d


def _copy_conv2dnorm(
    dst_conv2d: nn.Conv2d,
    dst_bn2d: nn.BatchNorm2d,
    src_conv2d: nn.Conv2d,
    src_bn2d: nn.BatchNorm2d,
) -> None:
    with torch.no_grad():
        dst_conv2d.weight.copy_(src_conv2d.weight)
        if dst_conv2d.bias is not None and src_conv2d.bias is not None:
            dst_conv2d.bias.copy_(src_conv2d.bias)

        dst_bn2d.weight.copy_(src_bn2d.weight)
        dst_bn2d.bias.copy_(src_bn2d.bias)

        if (
            dst_bn2d.running_mean is not None
            and src_bn2d.running_mean is not None
        ):
            dst_bn2d.running_mean.copy_(src_bn2d.running_mean)

        if (
            dst_bn2d.running_var is not None
            and src_bn2d.running_var is not None
        ):
            dst_bn2d.running_var.copy_(src_bn2d.running_var)

        if (
            dst_bn2d.num_batches_tracked is not None
            and src_bn2d.num_batches_tracked is not None
        ):
            dst_bn2d.num_batches_tracked.copy_(
                src_bn2d.num_batches_tracked
            )


def _duplicate_conv2dnorm(
    template: tuple[int, int, tuple[int, int], bool, bool],
    conv2d: nn.Conv2d,
    bn2d: nn.BatchNorm2d,
    device
) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
    rconv2d, rbn2d = _from_template(template, device)
    _copy_conv2dnorm(rconv2d, rbn2d, conv2d, bn2d)

    return rconv2d, rbn2d


def _finalize_conv2dnorm(
    template: tuple[int, int, tuple[int, int], bool, bool],
    conv2d: list[nn.Conv2d],
    bn2d: list[nn.BatchNorm2d],
    device
) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
    if len(conv2d) != len(bn2d):
        raise RuntimeError(f"length missmatch: {len(conv2d)} != {len(bn2d)}")

    rconv2d, rbn2d = _from_template(template, device)

    div = len(conv2d)
    with torch.no_grad():
        rconv2d.weight.copy_(torch.div(sum([c.weight for c in conv2d]), div))
        if rconv2d.bias is not None:
            rconv2d.bias.copy_(
                torch.div(
                    sum([c.bias for c in conv2d if c.bias is not None]), div
                )
            )

        rbn2d.weight.copy_(torch.div(sum([b.weight for b in bn2d]), div))
        rbn2d.bias.copy_(torch.div(sum([b.bias for b in bn2d]), div))

        if rbn2d.running_mean is not None:
            rbn2d.running_mean.copy_(
                torch.div(
                    sum([
                        b.running_mean
                        for b in bn2d if b.running_mean is not None
                    ]),
                    div
                )
            )

        if rbn2d.running_var is not None:
            rbn2d.running_var.copy_(
                torch.div(
                    sum([
                        b.running_var
                        for b in bn2d if b.running_var is not None
                    ]),
                    div
                )
            )

        if rbn2d.num_batches_tracked is not None:
            rbn2d.num_batches_tracked.copy_(
                torch.div(
                    sum([
                        b.num_batches_tracked
                        for b in bn2d if b.num_batches_tracked is not None
                    ]),
                    div,
                    rounding_mode="floor"
                )
            )

    return rconv2d, rbn2d


class _CommonWarmUpCtx(WarmUpCtx):
    _first_template: tuple[int, int, tuple[int, int], bool, bool]
    _first_conv2d: list[nn.Conv2d]
    _first_bn2d: list[nn.BatchNorm2d]

    _last_template: tuple[int, int, tuple[int, int], bool, bool]
    _last_conv2d: list[nn.Conv2d]
    _last_bn2d: list[nn.BatchNorm2d]

    def pre(self, supernet: SuperNet, device) -> None:
        first = supernet.first
        self._first_template = _to_template(first[0], first[1])
        self._first_conv2d = []
        self._first_bn2d = []

        last = supernet.last
        self._last_template = _to_template(last[0], last[1])
        self._last_conv2d = []
        self._last_bn2d = []

        self._device = device

    def _after_set(self, max_net: MobileSkeletonNet) -> None:
        first = max_net.first
        first_conv2d, first_bn2d = _duplicate_conv2dnorm(
            self._first_template, first[0], first[1], self._device
        )
        self._first_conv2d.append(first_conv2d)
        self._first_bn2d.append(first_bn2d)

        last = max_net.last
        last_conv2d, last_bn2d = _duplicate_conv2dnorm(
            self._last_template, last[0], last[1], self._device
        )
        self._last_conv2d.append(last_conv2d)
        self._last_bn2d.append(last_bn2d)

    def post(self, supernet: SuperNet) -> None:
        first_conv2d, first_bn2d = _finalize_conv2dnorm(
            self._first_template,
            self._first_conv2d,
            self._first_bn2d,
            self._device
        )
        first = supernet.first
        _copy_conv2dnorm(first[0], first[1], first_conv2d, first_bn2d)

        last_conv2d, last_bn2d = _finalize_conv2dnorm(
            self._last_template,
            self._last_conv2d,
            self._last_bn2d,
            self._device
        )
        last = supernet.last
        _copy_conv2dnorm(last[0], last[1], last_conv2d, last_bn2d)


def local_reorder_conv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.CONV2D][0], prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    return indices


def local_reorder_dwconv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], prev_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], prev_indices)

    indices = l1_reorder_main(src[LayerName.PWCONV2D][0], prev_indices)
    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], indices)

    return indices


def local_reorder_bdwrconv2d(
    src: BaseModule, prev_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.EXPANSION][0], prev_indices)
    if has_norm(src[LayerName.EXPANSION]):
        l1_reorder_batchnorm2d(src[LayerName.EXPANSION][1], indices)

    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    indices = l1_reorder_main(src[LayerName.PWCONV2D][0], indices)
    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], indices)

    return indices


def global_reorder_conv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 1, prev_indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.CONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.CONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], next_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], next_indices)

    return next_indices


def global_reorder_dwconv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, prev_indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], prev_indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], prev_indices)

    l1_reorder_conv2d(src[LayerName.PWCONV2D][0], 1, prev_indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.PWCONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.PWCONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], next_indices)

    return next_indices


def global_reorder_bdwrconv2d(
    src: BaseModule,
    prev_indices: torch.Tensor | None,
    next_indices: torch.Tensor | None
) -> torch.Tensor | None:
    indices = l1_reorder_main(src[LayerName.EXPANSION][0], prev_indices)
    if has_norm(src[LayerName.EXPANSION]):
        l1_reorder_batchnorm2d(src[LayerName.EXPANSION][1], indices)

    l1_reorder_conv2d(src[LayerName.CONV2D][0], 0, indices)
    if has_norm(src[LayerName.CONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.CONV2D][1], indices)

    if LayerName.SE in src:
        l1_reorder_se(src[LayerName.SE], indices)

    l1_reorder_conv2d(src[LayerName.PWCONV2D][0], 1, indices)
    with torch.no_grad():
        reordered = torch.index_select(
            src[LayerName.PWCONV2D][0].weight, dim=0, index=next_indices
        )

        src[LayerName.PWCONV2D][0].weight = nn.Parameter(reordered)

    if has_norm(src[LayerName.PWCONV2D]):
        l1_reorder_batchnorm2d(src[LayerName.PWCONV2D][1], next_indices)

    return next_indices


class WarmUpCtxLocalReorder(_CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[[BaseModule, torch.Tensor | None], torch.Tensor | None]
    ]

    def __init__(self) -> None:
        self._reorder = {
            ConvOp.CONV2D: local_reorder_conv2d,
            ConvOp.DWCONV2D: local_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: local_reorder_bdwrconv2d
        }

    def set(
        self,
        supernet: SuperNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op

        indices = None
        for sn_block, max_block in zip(supernet.blocks, max_net.blocks):
            modules: list[BaseModule] = []

            for module in max_block:
                indices = self._reorder[op](module, indices)
                modules.append(module)

            sn_block._weight_copy(modules)

        l1_reorder_conv2d(max_net.last[0], 1, indices)
        self._after_set(max_net)


class WarmUpCtxGlobalReorder(_CommonWarmUpCtx):
    _reorder: dict[
        ConvOp,
        Callable[
            [BaseModule, torch.Tensor | None, torch.Tensor | None],
            torch.Tensor | None
        ]
    ]
    _norms: list[tuple[Model, list[torch.Tensor]]]

    def __init__(self) -> None:
        self._reorder = {
            ConvOp.CONV2D: global_reorder_conv2d,
            ConvOp.DWCONV2D: global_reorder_dwconv2d,
            ConvOp.BDWRCONV2D: global_reorder_bdwrconv2d
        }
        self._norms = []

    def set(
        self,
        supernet: SuperNet,
        max_model: Model,
        max_net: MobileSkeletonNet
    ) -> None:
        op = max_model.blocks[0].layers[0].op
        if op == ConvOp.CONV2D:
            name = LayerName.CONV2D
        else:
            name = LayerName.PWCONV2D

        norms: list[torch.Tensor] = []
        for sn_block, max_block in zip(supernet.blocks, max_net.blocks):
            modules: list[BaseModule] = []

            for module in max_block:
                features = module[name]

                norms.append(l1_norm(features[0].weight))
                modules.append(module)

            sn_block._weight_copy(modules)

        self._norms.append((max_model, norms))
        self._after_set(max_net)

    def post(self, supernet: SuperNet) -> None:
        locals = [n[1] for n in self._norms]
        summed: list[torch.Tensor] = [sum(l) for l in zip(*locals)]
        norms: list[torch.Tensor] = [
            torch.sort(x, dim=0, descending=True)[1] for x in summed
        ]

        for (model, _), last in zip(self._norms, self._last_conv2d):
            op = model.blocks[0].layers[0].op
            supernet.set(model)

            indices = None
            for i, block in enumerate(supernet.blocks):
                for j, module in enumerate(block.get(copy=False)):
                    idx = i * max(LAYER_CHOICES) + j

                    indices = self._reorder[op](module, indices, norms[idx])

            l1_reorder_conv2d(last, 1, indices)

        super().post(supernet)
