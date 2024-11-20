from typing import cast
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation

from ..searchspace import Layer, ConvOp
from ..searchspace.layers import (
    squeeze_channels,
    BaseModule,
    LayerName,
    Conv2dModule,
    DWConv2dModule,
    BDWRConv2dModule
)
from .share import (
    SharedModule,
    SharedConv2d,
    SharedBatchNorm2d,
    SharedConv2dNormActivation,
    SharedSqueezeExcitation
)


class SharedBaseModule(SharedModule, nn.Module):
    in_channels: int
    out_channels: int
    se_ratio: float

    layers: nn.ModuleDict

    _active: list[nn.Module] | None
    _shortcut: bool | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        se_ratio: float
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se_ratio = se_ratio

        self._active = None
        self._shortcut = None

    def __getitem__(self, key: LayerName) -> nn.Module:
        return self.layers[key]

    def __contains__(self, key: LayerName) -> bool:
        return key in self.layers

    def _set_check(self, layer: Layer, op: ConvOp) -> None:
        if layer.op is not op:
            raise RuntimeError(
                "convolution operation missmatch: "
                f"expected {op}, got {layer.op}"
            )

        stride = self[LayerName.CONV2D][0].stride
        if layer.stride != stride:
            raise RuntimeError(
                "stride missmatch: "
                f"expected {stride}, got {layer.stride}"
            )

        if (
            layer.shortcut
            and (stride != (1, 1) or layer.in_channels != layer.out_channels)
        ):
            raise RuntimeError(
                "shortcut missmatch: "
                f"expected stride (1, 1) and in channels == out channels, "
                f"got {layer.stride} and {layer.in_channels} != {layer.out_channels}"
            )

    def unset(self) -> None:
        self._active = None
        self._shortcut = None

    def _weight_initialization(self) -> None:
        self[LayerName.CONV2D]._weight_initialization()
        self[LayerName.SE]._weight_initialization()

        if LayerName.EXPANSION in self:
            self[LayerName.EXPANSION]._weight_initialization()

        if LayerName.PWCONV2D in self:
            self[LayerName.PWCONV2D][0]._weight_initialization()
            self[LayerName.PWCONV2D][1]._weight_initialization()

    def _weight_copy(self, module: BaseModule) -> None:
        if (
            (isinstance(self, SharedConv2dModule)
                and not isinstance(module, Conv2dModule))
            or (isinstance(self, SharedDWConv2dModule)
                and not isinstance(module, DWConv2dModule))
            or (isinstance(self, SharedBDWRConv2dModule)
                and not isinstance(module, BDWRConv2dModule))
        ):
            raise RuntimeError(
                "module missmatch: "
                f"expected {type(self)}, got {type(module)}"
            )

        self[LayerName.CONV2D]._weight_copy(module[LayerName.CONV2D])
        self[LayerName.SE]._weight_copy(module[LayerName.SE])

        if LayerName.EXPANSION in self:
            self[LayerName.EXPANSION]._weight_copy(module[LayerName.EXPANSION])

        if LayerName.PWCONV2D in self:
            self[LayerName.PWCONV2D][0]._weight_copy(
                module[LayerName.PWCONV2D][0]
            )
            self[LayerName.PWCONV2D][1]._weight_copy(
                module[LayerName.PWCONV2D][1]
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._active is None or self._shortcut is None:
            raise RuntimeError("no active layer path was selected")

        output = input
        for op in self._active:
            output = op(output)

        if self._shortcut:
            output = input + output

        return output


class SharedConv2dModule(SharedBaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        if expansion_ratio != 1:
            raise RuntimeError(
                f"expansion ratio missmatch: expected 1, got {expansion_ratio}"
            )

        super().__init__(in_channels, out_channels, se_ratio)

        self.layers = nn.ModuleDict(
            [
                (
                    LayerName.CONV2D,
                    SharedConv2dNormActivation(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        activation_layer=activation_layer,
                        inplace=True
                    )
                ),
                (
                    LayerName.SE,
                    SharedSqueezeExcitation(
                        out_channels,
                        squeeze_channels(out_channels, se_ratio),
                        activation_layer
                    )
                )
            ]
        )

    def set(self, layer: Layer) -> None:
        self._set_check(layer, ConvOp.CONV2D)

        self[LayerName.CONV2D].set(layer.kernel_size, layer.in_channels, layer.out_channels)
        self._active = [self[LayerName.CONV2D]]

        if layer.se_ratio > 0.0:
            self[LayerName.SE].set(layer.out_channels, squeeze_channels(layer.out_channels, layer.se_ratio))
            self._active.append(self[LayerName.SE])

        self._shortcut = layer.shortcut

    def get(self, copy: bool = False) -> Conv2dModule:
        if self._active is None or self._shortcut is None:
            raise RuntimeError("no active layer path was selected")

        cna2d = cast(Conv2dNormActivation, self._active[0].get(copy))

        conv2d = cast(nn.Conv2d, cna2d[0])

        se_ratio = 0.0 if len(self._active) == 1 else self.se_ratio

        base = Conv2dModule(
            conv2d.in_channels,
            conv2d.out_channels,
            se_ratio,
            self._shortcut,
            conv2d.kernel_size,
            conv2d.stride,
            None,
            None
        )

        base.layers[LayerName.CONV2D] = cna2d
        if len(self._active) > 1:
            base.layers[LayerName.SE] = self._active[1].get(copy)

        return base


class SharedDWConv2dModule(SharedBaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        if expansion_ratio != 1:
            raise RuntimeError(
                f"expansion ratio missmatch: expected 1, got {expansion_ratio}"
            )

        super().__init__(in_channels, out_channels, se_ratio)

        self.layers = nn.ModuleDict(
            [
                (
                    LayerName.CONV2D,
                    SharedConv2dNormActivation(
                        in_channels,
                        in_channels,
                        kernel_size,
                        stride,
                        groups=in_channels,
                        activation_layer=activation_layer,
                        inplace=True
                    )
                ),
                (
                    LayerName.SE,
                    SharedSqueezeExcitation(
                        in_channels,
                        squeeze_channels(in_channels, se_ratio),
                        activation_layer
                    )
                ),
                (
                    LayerName.PWCONV2D,
                    nn.Sequential(
                        SharedConv2d(
                            in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False
                        ),
                        SharedBatchNorm2d(out_channels)
                    )
                )
            ]
        )

    def set(self, layer: Layer) -> None:
        self._set_check(layer, ConvOp.DWCONV2D)

        self[LayerName.CONV2D].set(layer.kernel_size, layer.in_channels, layer.in_channels)
        self._active = [self[LayerName.CONV2D]]

        if layer.se_ratio > 0.0:
            self[LayerName.SE].set(layer.in_channels, squeeze_channels(layer.in_channels, layer.se_ratio))
            self._active.append(self[LayerName.SE])

        pwconv2d = cast(SharedConv2d, self[LayerName.PWCONV2D][0])
        pwconv2d.set(pwconv2d.kernel_size, layer.in_channels, layer.out_channels)
        self[LayerName.PWCONV2D][1].set(layer.out_channels)
        self._active.append(self[LayerName.PWCONV2D])

        self._shortcut = layer.shortcut

    def get(self, copy: bool = False) -> DWConv2dModule:
        if self._active is None or self._shortcut is None:
            raise RuntimeError("no active layer path was selected")

        cna2d = cast(Conv2dNormActivation, self._active[0].get(copy))
        conv2d = cast(nn.Conv2d, cna2d[0])

        if len(self._active) < 3:
            se_ratio = 0.0
            pwidx = 1
        else:
            se_ratio = 0.25
            pwidx = 2

        pwseq = cast(nn.Sequential, self._active[pwidx])
        pwconv2d = cast(nn.Conv2d, pwseq[0].get(copy))

        base = DWConv2dModule(
            conv2d.in_channels,
            pwconv2d.out_channels,
            se_ratio,
            self._shortcut,
            conv2d.kernel_size,
            conv2d.stride,
            None,
            None
        )

        base.layers[LayerName.CONV2D] = cna2d
        if len(self._active) > 2:
            base.layers[LayerName.SE] = self._active[pwidx - 1].get(copy)

        base.layers[LayerName.PWCONV2D] = nn.Sequential(pwconv2d, pwseq[1].get(copy))

        return base


class SharedBDWRConv2dModule(SharedBaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        if expansion_ratio <= 1:
            raise RuntimeError(
                f"expansion ratio missmatch: expected > 1, got {expansion_ratio}"
            )

        super().__init__(in_channels, out_channels, se_ratio)

        hidden = in_channels * expansion_ratio

        self.layers = nn.ModuleDict(
            [
                (
                    LayerName.EXPANSION,
                    SharedConv2dNormActivation(
                        in_channels,
                        hidden,
                        kernel_size=1,
                        activation_layer=activation_layer,
                        inplace=True
                    )
                ),
                (
                    LayerName.CONV2D,
                    SharedConv2dNormActivation(
                        hidden,
                        hidden,
                        kernel_size,
                        stride,
                        groups=hidden,
                        activation_layer=activation_layer,
                        inplace=True
                    )
                ),
                (
                    LayerName.SE,
                    SharedSqueezeExcitation(
                        hidden,
                        squeeze_channels(hidden, se_ratio),
                        activation_layer
                    )
                ),
                (
                    LayerName.PWCONV2D,
                    nn.Sequential(
                        SharedConv2d(
                            hidden,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False
                        ),
                        SharedBatchNorm2d(out_channels)
                    )
                )
            ]
        )

    def set(self, layer: Layer) -> None:
        self._set_check(layer, ConvOp.BDWRCONV2D)

        hidden = layer.in_channels * layer.expansion_ratio

        expconv2d = cast(SharedConv2dNormActivation, self[LayerName.EXPANSION])
        expconv2d.set(expconv2d[0].kernel_size, layer.in_channels, hidden)
        self[LayerName.CONV2D].set(layer.kernel_size, hidden, hidden)
        self._active = [self[LayerName.EXPANSION], self[LayerName.CONV2D]]

        if layer.se_ratio > 0.0:
            self[LayerName.SE].set(hidden, squeeze_channels(hidden, layer.se_ratio))
            self._active.append(self[LayerName.SE])

        pwconv2d = cast(SharedConv2d, self[LayerName.PWCONV2D][0])
        pwconv2d.set(pwconv2d.kernel_size, hidden, layer.out_channels)
        self[LayerName.PWCONV2D][1].set(layer.out_channels)
        self._active.append(self[LayerName.PWCONV2D])

        self._shortcut = layer.shortcut

    def get(self, copy: bool = False) -> BDWRConv2dModule:
        if self._active is None or self._shortcut is None:
            raise RuntimeError("no active layer path was selected")

        ecna2d = cast(Conv2dNormActivation, self._active[0].get(copy))
        econv2d = cast(nn.Conv2d, ecna2d[0])

        cna2d = cast(Conv2dNormActivation, self._active[1].get(copy))
        conv2d = cast(nn.Conv2d, cna2d[0])

        if len(self._active) < 4:
            se_ratio = 0.0
            pwidx = 2
        else:
            se_ratio = 0.25
            pwidx = 3

        pwseq = cast(nn.Sequential, self._active[pwidx])
        pwconv2d = cast(nn.Conv2d, pwseq[0].get(copy))

        base = BDWRConv2dModule(
            econv2d.in_channels,
            pwconv2d.out_channels,
            conv2d.in_channels // econv2d.in_channels,
            se_ratio,
            self._shortcut,
            conv2d.kernel_size,
            conv2d.stride,
            None,
            None
        )

        base.layers[LayerName.EXPANSION] = ecna2d
        base.layers[LayerName.CONV2D] = cna2d
        if len(self._active) > 3:
            base.layers[LayerName.SE] = self._active[pwidx - 1].get(copy)

        base.layers[LayerName.PWCONV2D] = nn.Sequential(pwconv2d, pwseq[1].get(copy))

        return base


class SharedTrifectaModule(SharedModule, nn.Module):
    conv2d: SharedConv2dModule
    dwconv2d: SharedDWConv2dModule
    bdwrconv2d: SharedBDWRConv2dModule

    _active: SharedBaseModule | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int,
        se_ratio: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6
    ) -> None:
        super().__init__()

        self.conv2d = SharedConv2dModule(
            in_channels,
            out_channels,
            1,
            se_ratio,
            kernel_size,
            stride,
            activation_layer
        )

        self.dwconv2d = SharedDWConv2dModule(
            in_channels,
            out_channels,
            1,
            se_ratio,
            kernel_size,
            stride,
            activation_layer
        )

        self.bdwrconv2d = SharedBDWRConv2dModule(
            in_channels,
            out_channels,
            expansion_ratio,
            se_ratio,
            kernel_size,
            stride,
            activation_layer
        )

        self._active = None

    def set(self, layer: Layer) -> None:
        match layer.op:
            case ConvOp.CONV2D:
                self._active = self.conv2d
            case ConvOp.DWCONV2D:
                self._active = self.dwconv2d
            case ConvOp.BDWRCONV2D:
                self._active = self.bdwrconv2d
            case _:
                raise RuntimeError(
                    f"unknown convolution operation {layer.op}"
                )

        self._active.set(layer)

    def unset(self) -> None:
        if self._active is None:
            return

        self._active.unset()
        self._active = None

    def get(self, copy: bool = False) -> BaseModule:
        if self._active is None:
            raise RuntimeError("no active layer path was selected")

        return self._active.get(copy)

    def _weight_initialization(self) -> None:
        self.conv2d._weight_initialization()
        self.dwconv2d._weight_initialization()
        self.bdwrconv2d._weight_initialization()

    def _weight_copy(self, module: BaseModule) -> None:
        if isinstance(module, Conv2dModule):
            self.conv2d._weight_copy(module)
        elif isinstance(module, DWConv2dModule):
            self.dwconv2d._weight_copy(module)
        elif isinstance(module, BDWRConv2dModule):
            self.bdwrconv2d._weight_copy(module)
        else:
            raise RuntimeError(f"unknown module type {type(module)}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._active is None:
            raise RuntimeError("no active layer path was selected")

        return self._active(input)
