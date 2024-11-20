from typing import cast
from abc import abstractmethod, ABC
from collections.abc import Callable

import torch
import torch.nn as nn

from torchvision.ops.misc import (
    ConvNormActivation,
    Conv2dNormActivation,
    SqueezeExcitation
)


class SharedModule(ABC):
    @abstractmethod
    def set(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def unset(self) -> None:
        pass

    @abstractmethod
    def get(self, copy: bool = False) -> nn.Module:
        pass

    @abstractmethod
    def _weight_initialization(self) -> None:
        pass

    @abstractmethod
    def _weight_copy(self, module: nn.Module) -> None:
        pass


class SharedConv2d(SharedModule, nn.Conv2d):
    def set(self, kernel_size: int | tuple[int, int], in_channels: int, out_channels: int) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        size = self.weight.size()
        if size[0] < out_channels:
            raise RuntimeError(
                f"out channels missmatch: expected <= {size[0]}, got {out_channels}"
            )

        if size[1] < in_channels and out_channels != in_channels and self.groups != self.in_channels:
            raise RuntimeError(
                f"in channels missmatch: expected <= {size[1]}, got {in_channels}"
            )
        if size[2] < kernel_size[0]:
            raise RuntimeError(
                f"kernel height missmatch: expected <= {size[2]}, got {kernel_size[0]}"
            )
        if size[3] < kernel_size[1]:
            raise RuntimeError(
                f"kernel width missmatch: expected <= {size[3]}, got {kernel_size[1]}"
            )

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = tuple((ks - 1) // 2 * d for ks, d in zip(self.kernel_size, self.dilation))

        if self.groups != 1:
            self.groups = in_channels

    def unset(self) -> None:
        pass

    def get(self, copy: bool = False) -> nn.Conv2d:
        conv2d = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode
        )

        out_channels, in_channels, height, width = self._shared_indices()
        weight = self.weight[:out_channels, :in_channels, height[0]:height[1], width[0]:width[1]]

        if copy:
            with torch.no_grad():
                conv2d.weight.copy_(weight)

                if self.bias is not None:
                    conv2d.bias.copy_(self.bias[:out_channels])
        else:
            conv2d.weight = nn.Parameter(weight)

            if self.bias is not None:
                conv2d.bias = nn.Parameter(self.bias[:out_channels])

        return conv2d

    def _shared_indices(self) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
        _, _, weight_height, weight_width = self.weight.size()

        weight_height_center = int((weight_height - 1) / 2)
        weight_width_center = int((weight_width - 1) / 2)

        kernel_height, kernel_width = self.kernel_size

        height_center = int((kernel_height - 1) / 2)
        width_center = int((kernel_width - 1) / 2)

        h_start = weight_height_center - height_center
        h_end = weight_height_center + height_center + 1

        w_start = weight_width_center - width_center
        w_end = weight_width_center + width_center + 1

        out_channels = self.out_channels
        in_channels = int(self.in_channels / self.groups)

        return out_channels, in_channels, (h_start, h_end), (w_start, w_end)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out_channels, in_channels, height, width = self._shared_indices()

        weight = self.weight[:out_channels, :in_channels, height[0]:height[1], width[0]:width[1]]
        bias = self.bias[:out_channels] if self.bias is not None else None

        return self._conv_forward(input, weight, bias)

    def _weight_initialization(self) -> None:
        """
        Initialize the weight and bias with kaiming normal.
        """
        nn.init.kaiming_normal_(self.weight, mode="fan_out")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _weight_copy(self, src: nn.Conv2d) -> None:
        """
        Copy the weight and bias from a regular `torch.nn.Conv2d`.

        Args:
            src (torch.nn.Conv2d):
                The conv2d to copy the weights from.

        Raises:
            RuntimeError:
                On missmatch of the tensor sizes or bias present.
        """
        if self.weight.size() != src.weight.size():
            raise RuntimeError(
                "missmatch of weight size: "
                f"expected {self.weight.size()}, got {src.weight.size()}"
            )

        if (
            self.bias is not None and src.bias is not None
            and self.bias.size() != src.bias.size()
        ):
            raise RuntimeError(
                "missmatch of bias size: "
                f"expected {self.bias.size()}, got {src.bias.size()}"
            )

        if (
            self.bias is not None and src.bias is None
            or self.bias is None and src.bias is not None
        ):
            raise RuntimeError(
                "missmatch of bias present: "
                f"expected {self.bias is not None}, got {src.bias is not None}"
            )

        with torch.no_grad():
            self.weight.copy_(src.weight)
            if self.bias is not None:
                self.bias.copy_(src.bias)


class SharedBatchNorm2d(SharedModule, nn.BatchNorm2d):
    def set(self, num_features: int) -> None:
        size = self.weight.size()
        if size[0] < num_features:
            raise RuntimeError(
                f"num features missmatch: expected <= {size[0]}, got {num_features}"
            )

        self.num_features = num_features

    def unset(self) -> None:
        pass

    def get(self, copy: bool = False) -> nn.BatchNorm2d:
        bn2d = nn.BatchNorm2d(
            self.num_features,
            self.eps,
            self.momentum,
            self.affine,
            self.track_running_stats
        )

        weight = self.weight[:self.num_features]
        bias = self.bias[:self.num_features]

        if copy:
            with torch.no_grad():
                bn2d.weight.copy_(weight)
                bn2d.bias.copy_(bias)

                if self.running_mean is not None:
                    bn2d.running_mean.copy_(self.running_mean[:self.num_features])

                if self.running_var is not None:
                    bn2d.running_var.copy_(self.running_var[:self.num_features])

                if self.num_batches_tracked is not None:
                    bn2d.num_batches_tracked.copy_(self.num_batches_tracked)
        else:
            bn2d.weight = nn.Parameter(weight)
            bn2d.bias = nn.Parameter(bias)

            if self.running_mean is not None:
                bn2d.running_mean = self.running_mean[:self.num_features]

            if self.running_var is not None:
                bn2d.running_var = self.running_var[:self.num_features]

            if self.num_batches_tracked is not None:
                bn2d.num_batches_tracked = self.num_batches_tracked

        return bn2d

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        runnig_var = self.running_var

        self.weight = nn.Parameter(self.weight[:self.num_features])
        self.bias = nn.Parameter(self.bias[:self.num_features])

        if self.running_mean is not None:
            self.running_mean = self.running_mean[:self.num_features]

        if self.running_var is not None:
            self.running_var = self.running_var[:self.num_features]

        output = super().forward(input)

        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = runnig_var

        return output

    def _weight_initialization(self) -> None:
        """
        Initialize the weight and bias with kaiming normal.
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        if self.running_mean is not None:
            nn.init.zeros_(self.running_mean)

        if self.running_var is not None:
            nn.init.ones_(self.running_var)

        if self.num_batches_tracked is not None:
            nn.init.zeros_(self.num_batches_tracked)

    def _weight_copy(self, src: nn.BatchNorm2d) -> None:
        """
        Copy the weight, bias and running buffers from a regular
        `torch.nn.BatchNorm2d`.

        Args:
            src (torch.nn.BatchNorm2d):
                The batchnorm2d to copy the weights from.

        Raises:
            RuntimeError:
                On missmatch of the tensor sizes.
        """
        if self.weight.size() != src.weight.size():
            raise RuntimeError(
                "missmatch of weight size: "
                f"expected {self.weight.size()}, got {src.weight.size()}"
            )

        if self.bias.size() != src.bias.size():
            raise RuntimeError(
                "missmatch of bias size: "
                f"expected {self.bias.size()}, got {src.bias.size()}"
            )

        with torch.no_grad():
            self.weight.copy_(src.weight)
            self.bias.copy_(src.bias)

            src_running_mean = src.running_mean
            if self.running_mean is not None and src_running_mean is not None:
                self.running_mean.copy_(src_running_mean)

            src_running_var = src.running_var
            if self.running_var is not None and src_running_var is not None:
                self.running_var.copy_(src_running_var)

            src_num_batches_tracked = src.num_batches_tracked
            if (
                self.num_batches_tracked is not None
                and src_num_batches_tracked is not None
            ):
                self.num_batches_tracked.copy_(src_num_batches_tracked)


class SharedConv2dNormActivation(SharedModule, ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str | None = None,
        groups: int = 1,
        activation_layer: Callable[..., torch.nn.Module] | None = torch.nn.ReLU6,
        dilation: int | tuple[int, int] = 1,
        inplace: bool | None = True,
        bias: bool | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            SharedBatchNorm2d,
            activation_layer,
            dilation,
            inplace,
            bias,
            SharedConv2d,
        )

    def _extract(self) -> tuple[SharedConv2d, SharedConv2d]:
        sconv2d = cast(SharedConv2d, self[0])
        sbn2d = cast(SharedBatchNorm2d, self[1])

        return sconv2d, sbn2d

    def set(self, kernel_size: int | tuple[int, int], in_channels: int, out_channels: int) -> None:
        sconv2d, sbn2d = self._extract()

        sconv2d.set(kernel_size, in_channels, out_channels)
        sbn2d.set(out_channels)

    def unset(self) -> None:
        pass

    def get(self, copy: bool = False) -> Conv2dNormActivation:
        sconv2d = cast(SharedConv2d, self[0])
        sbn2d = cast(SharedBatchNorm2d, self[1])

        cna2d = Conv2dNormActivation(
            sconv2d.in_channels,
            sconv2d.out_channels,
            sconv2d.kernel_size,
            sconv2d.stride,
            sconv2d.padding,
            sconv2d.groups,
            None,
            None
        )

        cna2d.pop(0)
        cna2d.append(sconv2d.get(copy))
        cna2d.append(sbn2d.get(copy))
        cna2d.append(self[2])

        return cna2d

    def _weight_initialization(self) -> None:
        sconv2d, sbn2d = self._extract()

        sconv2d._weight_initialization()
        sbn2d._weight_initialization()

    def _weight_copy(self, src: Conv2dNormActivation) -> None:
        sconv2d, sbn2d = self._extract()

        sconv2d._weight_copy(src[0])
        sbn2d._weight_copy(src[1])


class SharedSqueezeExcitation(SharedModule, SqueezeExcitation):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU6,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        nn.Module.__init__(self)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = SharedConv2d(input_channels, squeeze_channels, 1)
        self.fc2 = SharedConv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def set(self, input_channels: int, squeeze_channels: int) -> None:
        self.fc1.set(self.fc1.kernel_size, input_channels, squeeze_channels)
        self.fc2.set(self.fc2.kernel_size, squeeze_channels, input_channels)

    def unset(self) -> None:
        pass

    def get(self, copy: bool = False) -> SqueezeExcitation:
        se = SqueezeExcitation(
            self.fc1.in_channels,
            self.fc1.out_channels,
            self.activation.__class__,
            self.scale_activation.__class__
        )

        se.fc1 = self.fc1.get(copy)
        se.fc2 = self.fc2.get(copy)

        return se

    def _weight_initialization(self) -> None:
        self.fc1._weight_initialization()
        self.fc2._weight_initialization()

    def _weight_copy(self, src: SqueezeExcitation) -> None:
        self.fc1._weight_copy(src.fc1)
        self.fc2._weight_copy(src.fc2)