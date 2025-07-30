import json
import argparse
from pathlib import Path
from shutil import rmtree
from logging import Logger
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from unas.oneshot import (
    supernet_train,
    initial_population,
    evolution,
    SuperNet,
    TrainCtx,
    EvolutionCtx,
    ChoiceBlockCtor,
    FitnessFn,
    MutatorFn
)
from unas.searchspace import Model, SearchSpace
from unas.datasets import CustomDataset
from unas.variants import full, conv2d, dwconv2d, bdwrconv2d

from unas.shim.runner import (
    memory_footprint,
    start_shim_runner,
    stop_shim_runner
)

from helper import *


def _save_supernet(
    path: Path, supernet: SuperNet, width_mult: float, resolution: int
) -> None:
    _mkdir(path)

    supernet.unset()

    weights_path = path / "weights.pth"
    weights = supernet.state_dict()

    with open(weights_path, "wb") as f:
        torch.save(weights, f)

    state = {
        "width-mult": width_mult,
        "resolution": resolution,
        "weights": str(weights_path.resolve())
    }

    with open(path / "state.json", "w") as f:
        json.dump(state, f)


def _mkdir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        path.mkdir(parents=True)


class SaveTrainCtx(TrainCtx):
    _path: Path
    _every: int
    _first: bool

    _width_mult: float
    _resolution: int

    def __init__(
        self,
        path: str | Path,
        every: int,
        first: bool = False,
        width_mult: float = 1.0,
        resolution: int = 224
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self._every = every
        self._first = first

        self._width_mult = width_mult
        self._resolution = resolution

        _mkdir(self._path)

    def epoch(self, epoch: int, supernet: SuperNet, epochs: int) -> None:
        if epoch % self._every != 0 and not (epoch == 1 and self._first):
            return

        path = self._path / f"{epoch}"
        _save_supernet(path, supernet, self._width_mult, self._resolution)


def _save_top(path: Path, k: int, top: list[tuple[Model, float]]) -> None:
    _mkdir(path)

    meta: dict[str, dict[str, str | float]] = {}
    for i, (model, accuracy) in enumerate(top[:k]):
        name = f"top-{i}"
        file = path / f"{name}.json"

        meta[name] = {"path": str(path.resolve()), "accuracy": accuracy}
        with open (file, "w") as f:
            json.dump(model.to_dict(), f)

    with open (path / "meta.json", "w") as f:
        json.dump(meta, f)


class SaveEvolutionCtx(EvolutionCtx):
    _path: Path
    _every: int
    _first: bool

    def __init__(
        self, path: str | Path, every: int, first: bool = False
    ) -> None:
        if isinstance(path, str):
            path = Path(path)

        self._path = path
        self._every = every
        self._first = first

        _mkdir(self._path)

    def iteration(
        self,
        iteration: int,
        population: list[Model],
        top: list[tuple[Model, float]]
    ) -> None:
        if (
            iteration % self._every != 0
            and not (iteration == 1 and self._first)
        ):
            return

        base = self._path / f"iteration-{iteration}"

        path_population = base / "population"
        _mkdir(path_population)
        for i, model in enumerate(population):
            with open(path_population / f"{i}.json", "w") as f:
                json.dump(model.to_dict(), f)

        path_top = base / f"top-{len(top)}"
        _save_top(path_top, len(top), top)


_spaces = {
    "full": (
        full.FullSearchSpace,
        full.full_choice_block_ctor,
        full.mutator
    ),
    "conv2d": (
        conv2d.Conv2dSearchSpace,
        conv2d.conv2d_choice_block_ctor,
        conv2d.mutator
    ),
    "dwconv2d": (
        dwconv2d.DWConv2dSearchSpace,
        dwconv2d.dwconv2d_choice_block_ctor,
        dwconv2d.mutator
    ),
    "bdwrconv2d": (
        bdwrconv2d.BDWRConv2dSearchSpace,
        bdwrconv2d.bdwrconv2d_choice_block_ctor,
        bdwrconv2d.mutator
    )
}


def _parse_search_space(
    args: argparse.Namespace
) -> tuple[SearchSpace, ChoiceBlockCtor, MutatorFn]:
    if args.space not in _spaces:
        raise ValueError(f"unknown search space {args.space}")

    space_ctor, choice_block_ctor, mutator = _spaces[args.space]
    return (
        space_ctor(args.width_mult, args.resolution),
        choice_block_ctor,
        mutator
    )


def _parse_state(args: argparse.Namespace) -> OrderedDict[str, torch.Tensor]:
    with open(Path(args.state), "rb") as f:
        state = torch.load(f, weights_only=True)

    return state


def _common(
    args: argparse.Namespace
) -> tuple[Path, CustomDataset, SearchSpace, SuperNet, Logger, MutatorFn]:
    path_results = Path(args.results)
    if not path_results.exists() or not path_results.is_dir():
        path_results.mkdir(parents=True)

    ds = ds_load(
        args.dataset, args.path, "train", args.ds_json, args.resolution
    )

    space, choice_block_ctor, mutator = _parse_search_space(args)
    supernet = SuperNet(
        ds.classes,
        args.width_mult,
        choice_block_ctor,
        initialize_weights="state" in args
    )

    if "state" in args:
        state = _parse_state(args)

        supernet.load_state_dict(state)

    logger = parse_logger(__name__, args.logger)

    return path_results, ds, space, supernet, logger, mutator


def _train(args: argparse.Namespace) -> None:
    path, ds, space, supernet, logger, _ = _common(args)

    if args.save_every > 0:
        train_ctx = SaveTrainCtx(
            path / "supernet" / "intermediate", args.save_every
        )
    else:
        train_ctx = None

    supernet_train(
        supernet,
        space,
        DataLoader(ds, **build_dl_config(args, args.batch_size)),
        args.epochs,
        args.models_per_batch,
        0.05,
        logger=logger,
        ctx=train_ctx,
        track_loss=3,
        device=torch.device(args.device)
    )
    _save_supernet(
        path / "supernet", supernet, args.width_mult, args.resolution
    )


def _fitness_wrapper(
    classes: int, resolution: int, max_flash: int, max_sram: int
) -> FitnessFn:
    def fitness(model: Model) -> bool:
        flash, sram = memory_footprint(model, classes, resolution)

        return flash <= max_flash and sram <= max_sram

    return fitness


def _evo(args: argparse.Namespace) -> None:
    path, ds, space, supernet, logger, mutator = _common(args)

    if args.save_every > 0:
        evo_ctx = SaveEvolutionCtx(
            path / "evolution" / "intermediate", args.save_every
        )
    else:
        evo_ctx = None

    start_shim_runner()

    try:
        max_flash = parse_memory(args.flash)
        max_sram = parse_memory(args.sram)
        fitness = _fitness_wrapper(
            ds.classes, args.resolution, max_flash, max_sram
        )

        population = initial_population(iter(space), args.population, fitness)

        top = evolution(
            supernet,
            DataLoader(ds, **build_dl_config(args, args.batch_size)),
            DataLoader(ds, **build_dl_config(args, args.calib_batch_size)),
            population,
            args.topk,
            args.iterations,
            args.population,
            mutator=mutator,
            fitness=fitness,
            logger=logger,
            ctx=evo_ctx,
            device=torch.device(args.device)
        )
    finally:
        stop_shim_runner()

    _save_top(path / "evolution", args.save_topk, top)


def _train_sub_parser(sub: argparse._SubParsersAction) -> None:
    train: argparse.ArgumentParser = sub.add_parser(
        "train",
        help="performs only the training of the supernet"
    )
    train.add_argument(
        "--save-every",
        help="save the intermediate state of the training every n epoch",
        type=int,
        default=0
    )
    train.add_argument(
        "--epochs",
        help="the number of epochs to train",
        type=int,
        default=120
    )
    train.add_argument(
        "--models-per-batch",
        help="the number of models sampled per batch of training data",
        type=int,
        default=1
    )
    train.add_argument(
        "results",
        help="path to store the results of the training",
        type=str
    )
    train.set_defaults(func=_train)


def _evo_sub_parser(sub: argparse._SubParsersAction) -> None:
    evo: argparse.ArgumentParser = sub.add_parser(
        "evo",
        help="performs only the evolution search"
    )
    evo.add_argument(
        "--flash",
        help="the flash limitation",
        type=str,
        default="inf"
    )
    evo.add_argument(
        "--sram",
        help="the sram limitation",
        type=str,
        default="inf"
    )
    evo.add_argument(
        "--topk",
        help="the topk models to keep",
        type=int,
        default=20
    )
    evo.add_argument(
        "--save-topk",
        help="save the first n from topk",
        type=int,
        default=5
    )
    evo.add_argument(
        "--calib-batch-size",
        help="the batch size used for calibration",
        type=int,
        default=64
    )
    evo.add_argument(
        "--population",
        help="the population size",
        type=int,
        default=100
    )
    evo.add_argument(
        "--iterations",
        help="the number of iterations",
        type=int,
        default=30
    )
    evo.add_argument(
        "--save-every",
        help="save the intermediate state of the evolution every n iteration",
        type=int,
        default=0
    )
    evo.add_argument(
        "state",
        help="path to the supernet weights to load",
        type=str
    )
    evo.add_argument(
        "results",
        help="path to store the results of the evolution search",
        type=str
    )
    evo.set_defaults(func=_evo)


def _args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    args_argument_logger(parser)
    args_argument_device(parser)

    parser.add_argument(
        "space",
        help="the search space to use",
        type=str,
        choices=list(_spaces.keys())
    )

    parser.add_argument(
        "--width-mult",
        required=True,
        help="width multiplier",
        type=float
    )

    args_group_ds(parser)

    sub = parser.add_subparsers()
    _train_sub_parser(sub)
    _evo_sub_parser(sub)

    return parser


def main() -> None:
    parser = _args_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
