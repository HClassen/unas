import re
import sys
import json
import argparse
from typing import Any
from collections.abc import Callable
from logging import (
    INFO,
    getLogger,
    FileHandler,
    StreamHandler,
    Formatter,
    Logger
)

import torch

from unas.datasets import transform_resize, CustomDataset, imagenet, gtsrb


__all__ = [
    "ds_load",
    "build_dl_config",
    "args_group_ds",
    "args_argument_logger",
    "args_argument_device",
    "parse_memory",
    "parse_logger"
]


def _gtsrb_load(
    path: str,
    split: str | None,
    ds_json: bool,
    transform: Callable[[torch.Tensor], torch.Tensor] | None
) -> gtsrb.GTSRBDataset:
    if ds_json:
        with open (path, "r") as f:
            config = json.load(f)
        return gtsrb.GTSRBDataset.from_dict(config, transform)

    return gtsrb.GTSRBDataset(path, split, transform)


def _ilsvrc_load(
    path: str,
    split: str | None,
    ds_json: bool,
    transform: Callable[[torch.Tensor], torch.Tensor] | None
) -> imagenet.ImageNetDataset:
    if ds_json:
        with open (path, "r") as f:
            config = json.load(f)
        return imagenet.ImageNetDataset.from_dict(config, transform)

    return imagenet.ImageNetDataset(path, split, transform)


_datasets = {
    "ilsvrc": _ilsvrc_load,
    "gtsrb": _gtsrb_load
}


def ds_load(
    name: str,
    path: str,
    split: str | None,
    ds_json: bool,
    resolution: int | None
) -> CustomDataset:
    if resolution is not None:
        transform = transform_resize(resolution)
    else:
        transform = None

    if name not in _datasets:
        raise ValueError(f"unknown data set {name}")

    return _datasets[name](path, split, ds_json, transform)


def build_dl_config(
    args: argparse.Namespace, batch_size: int
) -> dict[str, Any]:
    dl_config = {
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "shuffle": args.shuffle
    }

    if args.pin_memory:
        dl_config["pin_memory"] = args.pin_memory
        dl_config["pin_memory_device"] = args.device

    return dl_config


def args_group_ds(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "dataset",
        "options to configure the used data set"
    )

    group.add_argument(
        "dataset",
        help="the data set to use",
        type=str,
        choices=list(_datasets.keys())
    )
    group.add_argument(
        "--resolution",
        help="resolution of the images",
        required=True,
        type=int
    )
    group.add_argument(
        "--ds-json",
        help="load the data set from a JSON file instead of a folder",
        action="store_true"
    )
    group.add_argument(
        "--batch-size",
        help="data loader batch size",
        type=int,
        default=1024
    )
    group.add_argument(
        "--shuffle",
        help="shuffle the data set",
        action="store_true"
    )
    group.add_argument(
        "--num-workers",
        help="data loader workers",
        type=int,
        default=1
    )
    group.add_argument(
        "--pin-memory",
        help="data loader pin memory",
        action="store_true"
    )
    group.add_argument(
        "--prefetch-factor",
        help="data loader prefetch factor",
        type=int,
        default=1
    )
    group.add_argument(
        "path",
        help="the path to the data set",
        type=str
    )


def args_argument_logger(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-l", "--logger",
        help="path to store logging output or stdout",
        type=str,
        default="stdout"
    )

def args_argument_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d", "--device",
        help="torch device to run on",
        type=str,
        default="cpy"
    )


def parse_memory(memory: str) -> int:
    if memory == "inf":
        return sys.maxsize

    lower = memory.lower()
    match = re.match(r"^[0-9]+((k|m|g)?b)?$", lower)
    if match is None:
        raise RuntimeError(f"invalid memory size {memory}")

    if lower[-1].isdigit():
        return int(memory)

    unit_begin = -2 if lower[-2].isalpha() else -1

    value = lower[:unit_begin]
    unit = lower[unit_begin:]

    units = {"b": 1, "kb": 1_024, "mb": 1_024**2, "gb": 1_024**3}
    return int(value) * units[unit]


def parse_logger(name: str, dest: str) -> Logger:
    logger = getLogger(name)
    logger.setLevel(INFO)

    if "stdout" != dest:
        handler = FileHandler(dest)
    else:
        handler = StreamHandler(sys.stdout)

    formatter = Formatter("%(asctime)s::%(name)s::%(funcName)s::%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger