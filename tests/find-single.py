import re
import sys
import json
import argparse
from pathlib import Path
from collections.abc import Callable

from unas.searchspace import Model, SearchSpace
from unas.variants import full, conv2d, dwconv2d, bdwrconv2d
from unas.shim.runner import (
    memory_footprint,
    start_shim_runner,
    stop_shim_runner
)

from helper import parse_memory


_instances: dict[str, Callable[[float, int], SearchSpace]] = {
    "full": full.FullSearchSpace,
    "conv2d": conv2d.Conv2dSearchSpace,
    "dwconv2d": dwconv2d.DWConv2dSearchSpace,
    "bdwrconv2d": bdwrconv2d.BDWRConv2dSearchSpace
}


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width-mult",
        required=True,
        help="width multiplier",
        type=float
    )
    parser.add_argument(
        "--resolution",
        help="resolution of the images",
        required=True,
        type=int
    )
    parser.add_argument(
        "--classes",
        help="classes to learn",
        required=True,
        type=int
    )
    parser.add_argument(
        "--flash",
        help="the flash limitation",
        type=str,
        default="inf"
    )
    parser.add_argument(
        "--sram",
        help="the sram limitation",
        type=str,
        default="inf"
    )
    parser.add_argument(
        "space",
        help="select the search space to use",
        type=str,
        choices=["full", "conv2d", "dwconv2d", "bdwrconv2d"]
    )
    parser.add_argument(
        "result",
        help="path to store the result",
        type=str
    )

    return parser


def _save_model(model: Model, path: str) -> None:
    path: Path = Path(path)
    if path.parent.is_dir():
        path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "w") as f:
        json.dump(model.to_dict(), f)


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    ctor = _instances.get(args.space)
    print(ctor)
    if ctor is None:
        raise ValueError(f"unknown search space {args.space}")

    space = ctor(args.width_mult, args.resolution)
    max_flash = parse_memory(args.flash)
    max_sram = parse_memory(args.sram)

    start_shim_runner()
    try:
        for model in space:
            flash, sram = memory_footprint(model, args.classes, args.resolution)
            if flash <= max_flash and sram <= max_sram:
                _save_model(model, args.result)
                break
    finally:
        stop_shim_runner()


if __name__ == "__main__":
    main()
