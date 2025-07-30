import json
import argparse
from pathlib import Path
from logging import Logger
from itertools import islice

import torch
from torch.utils.data import DataLoader

from unas.mobilenet import skeletonnet_train, skeletonnet_valid, MobileSkeletonNet
from unas.datasets import transform_resize, imagenet
from unas.searchspace import Model
from unas.searchspace.model import build_model
from unas.oneshot import SuperNet, WarmUpCtx
from unas.variants import WarmUpCtxLocalReorder, WarmUpCtxGlobalReorder, full
from unas.utils import make_caption

from helper import args_argument_logger, parse_logger


WIDTH_MULT = 0.2
RESOLUTION = 48


def _sn_train_loop(
    sn: SuperNet, samples: list[Model], dl: DataLoader, logger: Logger, device
) -> None:
    for model in samples:
        sn.set(model)
        skeletonnet_train(sn, dl, 1, logger=logger, device=device)
        sn.unset()


def _sn_valid_loop(
    sn: SuperNet, samples: list[Model], dl: DataLoader, logger: Logger, device
) -> list[float]:
    accuracies = []
    for model in samples:
        sn.set(model)
        acc = skeletonnet_valid(sn, dl, logger=logger, device=device)
        sn.unset()

        accuracies.append(acc)

    return accuracies


def _warm_up(
    sn: SuperNet,
    ctx: WarmUpCtx,
    maxmodels: list[Model],
    maxnets: list[MobileSkeletonNet],
    device
) -> None:
    ctx.pre(sn, device)
    for mm, mn in zip(maxmodels, maxnets):
        ctx.set(sn, mm, mn)
    ctx.post(sn)


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    args_argument_logger(parser)

    parser.add_argument(
        "ds",
        help="path to the imagenet data set",
        type=str
    )

    parser.add_argument(
        "results",
        help="path to store the results",
        type=str
    )

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    ds = imagenet.ImageNetDataset(args.ds, "train", transform_resize(RESOLUTION))
    subds = imagenet.subset(ds, 50, 700)
    valid, train = imagenet.strict_split(subds, (50, ))

    space = full.FullSearchSpace(WIDTH_MULT, RESOLUTION)
    samples = list(islice(space, 100))

    path = Path(args.results)
    path.mkdir(parents=True, exist_ok=True)

    with open(path / "train.json", "w") as f:
        json.dump(train.to_dict(), f)

    with open(path / "valid.json", "w") as f:
        json.dump(valid.to_dict(), f)

    with open(path / "samples.json", "w") as f:
        json.dump([s.to_dict() for s in samples], f)

    logger = parse_logger(__name__, str(path / "weight-initialization.log"))
    device = torch.device("cuda:0")

    maxmodels = []
    maxnets = []
    accuracies = []
    train_dl = DataLoader(train, 128, True, num_workers=4, prefetch_factor=2)
    valid_dl = DataLoader(valid, 128, False, num_workers=4, prefetch_factor=2)
    for i, maxmodel in enumerate(space.max_models()):
        maxnet = build_model(maxmodel, train.classes, initialize_weights=True)

        skeletonnet_train(maxnet, train_dl, 100, logger=logger, device=device)
        acc = skeletonnet_valid(maxnet, valid_dl, device=device)

        accuracies.append(acc)
        maxmodels.append(maxmodel)
        maxnets.append(maxnet)

        with open(path / f"maxnet-{i}.pth", "wb") as f:
            torch.save(maxnet.state_dict(), f)

    logger.info(make_caption("Max Networks Accuracies", 70, "-"))
    for acc in accuracies:
        logger.info(f"accuracy: {acc:.4f}")

    logger.info(make_caption("SuperNet Kaiming", 70, "-"))
    sn = SuperNet(
        train.classes,
        WIDTH_MULT,
        full.full_choice_block_ctor,
        initialize_weights=True
    )
    _sn_train_loop(sn, samples, train_dl, logger, device)
    accuracies = _sn_valid_loop(sn, samples, valid_dl, device)
    with open(path / "kaiming.json", "w") as f:
        json.dump(accuracies, f)

    logger.log(make_caption("SuperNet Local Reorder", 70, "-"))
    sn = SuperNet(
        train.classes,
        WIDTH_MULT,
        full.full_choice_block_ctor,
        initialize_weights=False
    )
    sn.to(device)

    lctx = WarmUpCtxLocalReorder()
    _warm_up(sn, lctx, maxmodels, maxnets, device)
    _sn_train_loop(sn, samples, train_dl, logger, device)
    accuracies = _sn_valid_loop(sn, samples, valid_dl, device)
    with open(path / "local-reorder.json", "w") as f:
        json.dump(accuracies, f)

    logger.log(make_caption("SuperNet global Reorder", 70, "-"))
    sn = SuperNet(
        train.classes,
        WIDTH_MULT,
        full.full_choice_block_ctor,
        initialize_weights=False
    )
    sn.to(device)

    gctx = WarmUpCtxGlobalReorder()
    _warm_up(sn, gctx, maxmodels, maxnets, device)
    _sn_train_loop(sn, samples, train_dl, logger, device)
    accuracies = _sn_valid_loop(sn, samples, valid_dl, device)
    with open(path / "global-reorder.json", "w") as f:
        json.dump(accuracies, f)


if __name__ == "__main__":
    main()
