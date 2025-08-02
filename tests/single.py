import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from unas.searchspace import Model
from unas.searchspace.model import build_model
from unas.mobilenet import skeletonnet_train, skeletonnet_valid, MobileSkeletonNet

from helper import *


def _common(
    args: argparse.Namespace, split: str, initialize_weights: bool
) -> tuple[MobileSkeletonNet, DataLoader, torch.device, Logger]:
    ds = ds_load(args.dataset, args.path, split, args.ds_json, args.resolution)

    with open(args.model, "r") as f:
        model = Model.from_dict(json.load(f))

    net = build_model(
        model, ds.classes, initialize_weights=initialize_weights
    )

    dl = DataLoader(ds, **build_dl_config(args, args.batch_size))

    logger = parse_logger(__name__, args.logger)

    return net, dl, torch.device(args.device), logger


def _train(args: argparse.Namespace) -> None:
    net, dl, device, logger = _common(args, "train", args.weights is None)
    if args.weights is not None:
        with open(args.weights, "rb") as f:
            weights = torch.load(f, weights_only=True)

        net.load_state_dict(weights)

    skeletonnet_train(net, dl, args.epochs, logger=logger, device=device)

    path = Path(args.results)
    parent = path.parent
    if not parent.is_dir():
        parent.mkdir(parents=True)

    with open(path, "wb") as f:
        torch.save(net.state_dict(), f)


def _train_sub_parser(sub: argparse._SubParsersAction) -> None:
    train: argparse.ArgumentParser = sub.add_parser(
        "train",
        help="train a model on a data set"
    )
    train.add_argument(
        "--epochs",
        help="the number of epochs to train",
        type=int,
        default=120
    )
    train.add_argument(
        "--weights",
        help="load exisitng weights",
        type=str
    )
    train.add_argument(
        "model",
        help="path to the model definition",
        type=str
    )
    train.add_argument(
        "results",
        help="path to store the results of the training",
        type=str
    )
    train.set_defaults(func=_train)


def _valid(args: argparse.Namespace) -> None:
    net, dl, device, logger = _common(args, "test", False)
    with open(args.weights, "rb") as f:
        weights = torch.load(f, weights_only=True)

    net.load_state_dict(weights)
    acc = skeletonnet_valid(net, dl, logger=logger, device=device)

    logger.info(f"accuracy: {acc}")


def _valid_sub_parser(sub: argparse._SubParsersAction) -> None:
    valid: argparse.ArgumentParser = sub.add_parser(
        "valid",
        help="validate a model on a data set"
    )
    valid.add_argument(
        "model",
        help="path to the model definition",
        type=str
    )
    valid.add_argument(
        "weights",
        help="path to the model weights",
        type=str
    )
    valid.set_defaults(func=_valid)


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    args_argument_logger(parser)
    args_argument_device(parser)
    args_group_ds(parser)

    sub = parser.add_subparsers()
    _train_sub_parser(sub)
    _valid_sub_parser(sub)

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()