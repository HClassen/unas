import time
from copy import deepcopy
from logging import Logger
from itertools import islice
from typing import cast, Final
from bisect import insort_left
from abc import abstractmethod, ABC
from collections.abc import Callable, Iterable, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.ops import Conv2dNormActivation

from ..mobilenet import (
    FIRST_CONV_CHANNELS,
    LAYER_SETTINGS,
    build_first,
    build_pool,
    skeletonnet_train,
    skeletonnet_valid,
    MobileSkeletonNet
)
from ..searchspace import Layer, Block, Model, SearchSpace
from ..searchspace.layers import module_to_layer, BaseModule
from ..searchspace.model import build_model
from ..utils import make_divisible, make_caption

from .share import SharedModule
from .layers import SharedConv2dNormActivation


__all__ = [
    "OneShotNet",
    "ChoiceBlock",
    "ChoiceBlockCtor",
    "SuperNet",
    "WarmUpCtx",
    "supernet_warm_up",
    "supernet_train",
    "FitnessFn",
    "MutatorFn",
    "initial_population",
    "crossover",
    "mutation",
    "evaluate",
    "EvolutionCtx",
    "evolution"
]


class OneShotNet(MobileSkeletonNet):
    _width_mult: float
    _block_lengths: list[int]

    def __init__(
        self,
        first: nn.Module,
        blocks: Iterable[nn.Module],
        last: nn.Module,
        pool: nn.Sequential,
        width_mult: float
    ) -> None:
        super().__init__(first, blocks, last, pool, False)

        self._width_mult = width_mult

    def to_model(self) -> Model:
        """
        Build a `Model` from this `OneShotNet`.

        Returns:
            Model:
                The configuration of this network.
        """

        blocks: list[Block] = []
        for b in self.blocks:
            blocks.append(Block([module_to_layer(module) for module in b]))

        return Model(self._width_mult, blocks)


class ChoiceBlock(SharedModule, nn.Module):
    active_layers: int | None
    layers: nn.ModuleList  # An array of `SharedModule`

    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        max_expansion_ratio: int,
        max_se_ration: float,
        max_kernel_size: int | tuple[int, int],
        max_depth: int,
        stride: int,
        activation_layer: Callable[..., nn.Module] | None,
        shared_ctor: Callable[..., SharedModule]
    ) -> None:
        super().__init__()

        layers: list[SharedModule] = [
            shared_ctor(
                max_in_channels,
                max_out_channels,
                max_expansion_ratio,
                max_se_ration,
                max_kernel_size,
                stride,
                activation_layer
            )
        ] + [
            shared_ctor(
                max_out_channels,
                max_out_channels,
                max_expansion_ratio,
                max_se_ration,
                max_kernel_size,
                1,
                activation_layer
            ) for _ in range(max_depth - 1)
        ]

        self.active_layers = None
        self.layers = nn.ModuleList(layers)

    def set(self, block: Block) -> None:
        if len(block.layers) > len(self.layers):
            raise RuntimeError(
                "block length missmatch: "
                f"expected <= {len(self.layers)}, got {len(block.layers)}"
            )

        self.active_layers = None

        for choice_layer, block_layer in zip(self.layers, block.layers):
            choice_layer = cast(SharedModule, choice_layer)

            choice_layer.set(block_layer)

        self.active_layers = len(block.layers)

    def unset(self) -> None:
        for layer in self.layers[:self.active_layers]:
            layer = cast(SharedModule, layer)

            layer.unset()

        self.active_layers = None

    def get(self, copy: bool = False) -> list[BaseModule]:
        if self.active_layers is None:
            raise RuntimeError("no active block path was selected")

        return [
            layer.get(copy) for layer in self.layers[:self.active_layers]
        ]

    def _weight_initialization(self) -> None:
        for layer in self.layers:
            layer = cast(SharedModule, layer)

            layer._weight_initialization()

    def _weight_copy(self, modules: list[BaseModule]) -> None:
        for layer, module in zip(self.layers, modules):
            layer = cast(SharedModule, layer)

            layer._weight_copy(module)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active_layers is None:
            raise RuntimeError("no active block path was selected")

        for layer in self.layers[:self.active_layers]:
            inputs = layer(inputs)

        return inputs


ChoiceBlockCtor = Callable[
    [int, int, float, int, int, Callable[..., nn.Module] | None],
    tuple[int, ChoiceBlock]
]

class SuperNet(MobileSkeletonNet):
    first: Conv2dNormActivation
    blocks: list[ChoiceBlock]
    last: SharedConv2dNormActivation

    _width_mult: float

    def __init__(
        self,
        classes: int,
        width_mult: float,
        choice_block_ctor: ChoiceBlockCtor,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU6,
        initialize_weights: bool = True
    ) -> None:
        self._width_mult = width_mult

        round_nearest = 8
        in_channels = make_divisible(FIRST_CONV_CHANNELS * width_mult, round_nearest)
        first = build_first(in_channels, nn.BatchNorm2d, activation_layer)

        blocks: list[ChoiceBlock] = []
        for setting in LAYER_SETTINGS:
            out_channels, block = choice_block_ctor(
                in_channels,
                setting[1],
                width_mult,
                round_nearest,
                setting[3],
                activation_layer
            )

            blocks.append(block)
            in_channels = out_channels

        last = SharedConv2dNormActivation(
            in_channels, classes, 1, activation_layer=activation_layer
        )
        pool = build_pool()

        super().__init__(first, blocks, last, pool, initialize_weights)

    def _weight_initialization(self) -> None:
        nn.init.kaiming_normal_(self.first[0].weight, mode="fan_out")
        nn.init.ones_(self.first[1].weight)
        nn.init.zeros_(self.first[1].bias)

        for block in self.blocks:
            block._weight_initialization()

        self.last._weight_initialization()

    def set(self, model: Model) -> None:
        if len(model.blocks) > len(self.blocks):
            raise RuntimeError(
                "model length missmatch: "
                f"expected <= {len(self.blocks)}, got {len(model.blocks)}"
            )

        for choice_block, model_block in zip(self.blocks, model.blocks):
            choice_block = cast(ChoiceBlock, choice_block)

            choice_block.set(model_block)

        last = cast(nn.Conv2d, self.last[0])
        self.last.set(
            last.kernel_size,
            model.blocks[-1].layers[-1].out_channels,
            last.out_channels
        )

    def get(self, copy: bool = False) -> OneShotNet:
        first = self.first if not copy else deepcopy(self.first)

        blocks: list[nn.Sequential] = []
        for block in self.blocks:
            blocks.append(nn.Sequential(*block.get(copy)))

        last = self.last.get(copy)

        return OneShotNet(first, blocks, last, self.pool, self._width_mult)

    def unset(self) -> None:
        for block in self.blocks:
            block.unset()


class WarmUpCtx(ABC):
    @abstractmethod
    def set(
        self, supernet: SuperNet, max_model: Model, max_net: MobileSkeletonNet
    ) -> None:
        """
        Called to make use of the trained weights in a maximum model.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
            max_model (Model):
                The configuration of the current maximum model of the search
                space.
            max_net (MobileSkeletonNet):
                The actual Pytorch network of `max_model` with the trained
                weights.
        """
        pass

    def pre(self, supernet: SuperNet, device=None) -> None:
        """
        Called before the first maximum model is trained. This is optional.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
            device:
                The Pytorch device the network will be moved to.
        """
        pass

    def post(self, supernet: SuperNet) -> None:
        """
        Called after the last maximum model is trained. This is optional.

        Args:
            supernet (SuperNet):
                The super-network currently in warm up.
        """
        pass


def supernet_warm_up(
    supernet: SuperNet,
    space: SearchSpace,
    ctx: WarmUpCtx,
    dl: DataLoader,
    epochs: int,
    *,
    logger: Logger,
    batches: int | None = None,
    device=None
) -> None:
    """
    Warm up a `SuperNet` by training the maximum model(s) from the search
    space. Set the weights of the super-netowrk to the trained weights from
    the maximum model(s).

    Args:
        supernet (SuperNet):
            The super-network to warm up.
        space (SearchSpace):
            The search space to get the maximum models.
        ctx (WarmUpCtx):
            The warm up context.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set use for the warm up training.
        epochs (int):
            The number of training epochs.
        batch_size (int):
            The number of samples per batch.
        logger (Logger):
            The logger to output training status messages.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        device:
            The Pytorch device the network is moved to.
    """
    warm_up_start = time.time()

    ctx.pre(supernet, device=device)

    logger.info(make_caption("Warm Up", 70, " "))
    for i, max_model in enumerate(space.max_models()):
        net = build_model(
            max_model, supernet.last.out_channels
        )

        skeletonnet_train(
            net,
            dl,
            epochs,
            0.9,
            5e-5,
            logger=logger,
            batches=batches,
            device=device
        )

        logger.info(make_caption(f"Set Parameters({i + 1})", 70, "-"))

        set_start = time.time()
        ctx.set(supernet, max_model, net)
        set_time = time.time() - set_start

        logger.info(f"time={set_time:.2f}s")

    ctx.post(supernet)

    warm_up_time = time.time() - warm_up_start
    logger.info(f"total={warm_up_time:.2f}s")


class TrainCtx(ABC):
    @abstractmethod
    def epoch(self, epoch: int, supernet: SuperNet, epochs: int) -> None:
        """
        Gets called at the end of every epoch during training.

        Args:
            epoch (int):
                The current epoch.
            supernet (SuperNet):
                The super-network as is after a training epoch.
            epochs (int):
                The total number of epochs.
        """
        pass


def supernet_train(
    supernet: SuperNet,
    space: SearchSpace,
    dl: DataLoader,
    epochs: int,
    models_per_batch: int,
    initial_lr: float,
    momentum: float = 0.9,
    weight_decay: float = 5e-5,
    *,
    logger: Logger,
    ctx: TrainCtx | None = None,
    batches: int | None = None,
    track_loss: int | list[Model] | None = 0,
    device=None
) -> None:
    """
    Train a `SuperNet` by sampling random models per batch of training data,
    setting the super-network to the sampled model and performing the backward
    pass.

    Args:
        supernet (SuperNet):
            The super-network to train.
        space (SearchSpace):
            The search space from which to sample the models.
        dl (torch.utils.data.DataLoader):
            The data loader of the data set to train on.
        epochs (int):
            The number of training epochs.
        models_per_batch (int):
            The number of models sampled per batch.
        initial_lr (float):
            The initial learning rate for cosine annealing learning.
        momentum (float):
            The momenum for `SGD` to use.
        weight_decay (float):
            The weight decay for `SGD` to use.
        logger (Logger):
            The logger to output training status messages.
        ctx (TrainCtx, None):
            The train context.
        batches (int, None):
            The number of batches per epoch. If set to `None` use the whole data
            set.
        track_loss: (int, list[Model], None):
            Track the training progress by sampling a fixed size of models and
            get their loss after each epoch. Either sample `int` many models or
            provide a list of models.
        device:
            The Pytorch device the network is moved to.
    """
    train_start = time.time()

    models = iter(space)
    batches = batches if batches is not None else len(dl)

    supernet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        supernet.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    if track_loss is None:
        track_loss = 0

    if isinstance(track_loss, int):
        fixed_models = [model for model in islice(space, track_loss)]
    else:
        fixed_models = track_loss

    logger.info(make_caption("Training", 70, " "))
    supernet.train()
    for i in range(epochs):
        logger.info(make_caption(f"Epoch {i + 1}/{epochs}", 70, "-"))

        epoch_start = time.time()

        batch_start = time.time()  # to capture data load time
        for k, (images, labels) in enumerate(dl):
            lr = scheduler.get_last_lr()[0]

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            for _ in range(models_per_batch):
                model = next(models)
                supernet.set(model)

                outputs = supernet(images)
                loss = criterion(outputs, labels) / models_per_batch
                loss.backward()

            optimizer.step()

            batch_time = time.time() - batch_start

            logger.info(
                f"epoch={i + 1}, batch={k + 1:0{len(str(batches))}}/{batches}, lr={lr:.05f}, time={batch_time:.2f}s"
            )
            batch_start = time.time()

            if batches == k + 1:
                break

        scheduler.step()

        logger.info("\n")
        images, labels = next(iter(dl))
        images = images.to(device)
        labels = labels.to(device)
        for i, model in enumerate(fixed_models):
            supernet.set(model)
            outputs = supernet(images)
            loss = criterion(outputs, labels)
            logger.info(f"fixed-subnet-{i + 1}-loss={loss.item()}")

        supernet.unset()
        if ctx is not None:
            ctx.epoch(i + 1, supernet, epochs)

        epoch_time = time.time() - epoch_start
        logger.info(f"time={epoch_time:.2f}s")

    train_time = time.time() - train_start
    logger.info(f"total={train_time:.2f}s")



FitnessFn = Callable[[Model], bool]
MutatorFn = Callable[[int, float, Block], Block]


def initial_population(
    space: Iterator[Model],
    size: int,
    fitness: Callable[[Model], bool],
    limit: int = -1
) -> list[Model]:
    """
    Select the initial population of models for evolution search.

    Args:
        space (Iterator[Model]):
            An iterator over the search space, which (randomly) samples models.
        size (int):
            The initial population size.
        fitness (Callable[[Model], bool]):
            A function to evaluate the fitness of a model for some criterion
            but not accuracy.
        limit (int):
            A limit to the iterations of the generation loop. Default value
            `-1` means no limit. If the limit is reached, an exception is
            raised containing the sampled population as second argument.

    Returns:
        list[Model]:
            The selected population.

    Raises:
        RuntimeError:
            On limit for iterations reached.
    """
    iteration = 0
    population: list[OneShotNet] = []

    while len(population) < size:
        if limit != -1 and limit > iteration:
            raise RuntimeError(f"iteration limit reached: {limit}", population)

        model = next(space)

        if fitness(model):
            population.append(model)

        iteration += 1

    return population


def _adjust_block_border(prev: Block, next: Block) -> None:
    first = next.layers[0]
    in_channels = prev.layers[-1].out_channels

    if first.in_channels != in_channels:
        next.layers[0] = Layer(
            first.op,
            in_channels,
            first.out_channels,
            first.kernel_size,
            first.stride,
            first.expansion_ratio,
            False,
            first.se_ratio
        )


def _combine(
    width_mult: float,
    top: list[Block],
    bottom: list[Block]
) -> Model:
    _adjust_block_border(top[-1], bottom[0])

    blocks = top + bottom
    return Model(width_mult, blocks)


def crossover(p1: Model, p2: Model) -> tuple[Model, Model]:
    """
    Performs single point crossover on `p1` and `p1`. Randomly select a split
    point `s` of the seven inner blocks. Then select the blocks `0` to `s` from
    `p1` and merge them with the blocks `s + 1` to `6` from `p2` to create the
    first offspring model. The second offspring is produces by flipping `p1` and
    `p2`. The split point `s` is chosen between [1, min_block_length - 1].

    At the merge point for both models, set the `in_channels` of the bottom
    half to the `out_channels` of the top half.

    Args:
        p1 (Model):
            The first parent.
        p2 (Model):
            The second parent.

    Returns:
        tuple[Model, Model]:
            The two offsprings.
    """
    l = min(len(p1.blocks), len(p2.blocks))
    split = torch.randint(1, l, [1]).item()

    p1_top = deepcopy(p1.blocks[:split])
    p1_bottom = deepcopy(p1.blocks[split:])

    p2_top = deepcopy(p2.blocks[:split])
    p2_bottom = deepcopy(p2.blocks[split:])

    o1 = _combine(p1.width_mult, p1_top, p2_bottom)
    o2 = _combine(p2.width_mult, p2_top, p1_bottom)

    return o1, o2


def mutation(
    model: Model, p: float, mutator: MutatorFn
) -> Model:
    """
    Mutates the structure of `Model`. Each block of the model is passed to the
    supplied `mutator` with a propability of `p`.

    Args:
        model (Model):
            The model to mutate.
        p (float):
            The mutation probability.
        mutator (MutatorFn):
            Mutates a `Block` and returns it. It receives the index of the block,
            the width multiplicator of the model and the block itself.

    Returns:
        Model:
            The new and mutated model.
    """
    mutated = deepcopy(model)
    blocks: list[Block] = []
    for i, block in enumerate(model.blocks):
        copied = deepcopy(block)

        if torch.rand(1).item() < p:
            copied = mutator(i, mutated.width_mult, copied)

        blocks.append(copied)

    for i in range(1, len(blocks)):
        _adjust_block_border(blocks[i - 1], blocks[i])

    return Model(model.width_mult, blocks)


def evaluate(
    supernet: SuperNet,
    population: Iterable[Model],
    valid_dl: DataLoader,
    calib_dl: DataLoader,
    *,
    logger: Logger,
    valid_batches: int | None = None,
    calib_batches: int = 20,
    device=None
) -> list[float]:
    """
    Evaluates the candidates of a population regarding their accuracy. Each
    model inherits its weight from the `supernet`. Before evaluating a model,
    the running mean and var of its batch-normalization recalibrated using
    the `calib_dl`.

    Args:
        supernet (SuperNet):
            A (pre-trained) supernet to inherit weights from.
        population (Iterable[Model]):
            The population of one iteration in evolution search.
        valid_dl (torch.utils.data.DataLoader):
            The data loader of the data set used to evaluate the accuracy of a
            model.
        calib_dl (torch.utils.data.DataLoader):
            The data laoder of the data set used to recalibrate the
        logger (Logger):
            The logger to output training status messages.
            batch-normalizations of a model.
        valid_batches (int, None):
            The number of batches per epoch used during validation. If set to
            `None` use the whole data set.
        calib_batches (int):
            The number of batches used for recalibration.
        device:
            The Pytorch device the network is moved to.

    Returns:
        list[float]:
            The accuracies of the model in `population` in the same order.
    """
    results: list[float] = []

    for candidate in population:
        supernet.set(candidate)
        net = supernet.get(True)

        net.to(device)

        # Recalibrate batch norm statistics (running mean/var).
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

        with torch.no_grad():
            net.train()
            for (images, _) in islice(calib_dl, calib_batches):
                images = images.to(device)
                net(images)

        accuracy = skeletonnet_valid(
            net, valid_dl, logger=logger, batches=valid_batches, device=device
        )
        results.append(accuracy)
        supernet.unset()

    return results


class EvolutionCtx(ABC):
    @abstractmethod
    def iteration(
        self,
        iteration: int,
        population: list[Model],
        top: list[tuple[Model, float]]
    ) -> None:
        pass


def _add_to_top(
    top: list[tuple[Model, float]],
    population: list[Model],
    accuracies: list[float]
) -> None:
    for candidate, accuracy in zip(population, accuracies):
        insort_left(
            top, (candidate, accuracy), key=lambda entry: -entry[1]
        )


def _do_evaluate(
    supernet: SuperNet,
    population: Iterable[Model],
    valid_dl: DataLoader,
    calib_dl: DataLoader,
    valid_batches: int,
    calib_batches: int,
    top: list[tuple[Model, float]],
    topk: int,
    iteration: int,
    logger: Logger,
    device
) -> list[Model]:
    test_start = time.time()
    accuracies = evaluate(
        supernet,
        population,
        valid_dl,
        calib_dl,
        valid_batches=valid_batches,
        calib_batches=calib_batches,
        logger=logger,
        device=device
    )
    test_time = time.time() - test_start

    logger.info(
        f"iteration={iteration + 1}, method=evaluate, time={test_time:.2f}s"
    )

    _add_to_top(top, population, accuracies)
    return top[:topk]


def _rnd_from_top(top: list[tuple[Model, float]], topk: int) -> Model:
    return top[torch.randint(0, topk, (1,)).item()][0]


def _do_crossover(
    top: list[tuple[Model, float]],
    topk: int,
    n_cross: int,
    fitness: Callable[[Model], bool],
    iteration: int,
    logger: Logger
) -> list[Model]:
    crossover_start = time.time()

    cross: list[Model] = []
    while len(cross) < n_cross:
        p1 = _rnd_from_top(top, topk)
        p2 = _rnd_from_top(top, topk)

        c1, c2 = crossover(p1, p2)

        if fitness(c1):
            cross.append(c1)

        if fitness(c2):
            cross.append(c2)

    # If n_cross % 2 = 1 then adding two models per iteration results
    # in len(cross) == n_cross + 1.
    cross = cross[:n_cross]

    crossover_time = time.time() - crossover_start
    logger.info(
        f"iteration={iteration + 1}, method=crossover, time={crossover_time:.2f}s"
    )

    return cross


def _do_mutate(
    top: list[tuple[Model, float]],
    topk: int,
    n_mut: int,
    mutation_rate: float,
    mutator: MutatorFn,
    fitness: Callable[[Model], bool],
    iteration: int,
    logger: Logger
) -> list[Model]:
    mutate_start = time.time()

    mut: list[Model] = []
    while len(mut) != n_mut:
        mutated = mutation(_rnd_from_top(top, topk), mutation_rate, mutator)

        if fitness(mutated):
            mut.append(mutated)

    mutate_time = time.time() - mutate_start
    logger.info(
        f"iteration={iteration + 1}, method=mutate, time={mutate_time:.2f}s"
    )

    return mut


def evolution(
    supernet: SuperNet,
    valid_dl: DataLoader,
    calib_dl: DataLoader,
    initial_population: Iterable[Model],
    topk: int,
    iterations: int,
    population_size: int,
    next_gen_split: tuple[float, float] = (0.5, 0.5),
    mutation_rate: float = 0.1,
    mutator: MutatorFn = lambda a, b, c: c,
    fitness: Callable[[Model], bool] = lambda _: True,
    *,
    logger: Logger,
    ctx: EvolutionCtx | None = None,
    valid_batches: int | None = None,
    calib_batches: int | None = None,
    device=None
) -> list[tuple[Model, float]]:
    """
    Perform evolution search on the `SuperNet` to find the model with the
    highest accuracy. Per iteration, single-point crossover and mutation is
    performed on the current population. Before the accuracy of a model is
    tested, calibrate the batch-normalisation layers on the validation
    data.

    Args:
        supernet (SuperNet):
            The (pre-trainend) super-network to use as basis for the
            individual networks.
        valid_dl (torch.utils.data.DataLoader):
            The data loader of the data set used to evaluate the accuracy
            of a model.
        calib_dl (torch.utils.data.DataLoader):
            The data loader of the data set used to calibrate the
            batch-normalisation layers of a model.
        initial_population (Iterable[Model]):
            The initial population to start the evolution from.
        topk (int):
            The number of models with the highest accuracy to keep track
            off.
        iterations (int):
            For how many generations the evaluation search should run.
        population_size (int):
            The size of the population per generation.
        next_gen_split (tuple[float, float]):
            The percentages the next population is made up for from
            crossover and mutation.
        mutation_rate (float):
            Threashold for mutation to take effect.
        mutator (MutatorFn):
            The mutator function passed to the `mutation` function of the
            evolution search.
        fitness (Callable[[Model], bool]):
            Evaluate the fitness of a sampled model on some criteria. Only
            models where `fitness` returns `True` are selected for the
            population of the next generation.
        logger (Logger):
            The logger to output training status messages.
        ctx (EvolutionCtx, None):
            The evolution context.
        valid_batches (int, None):
            The number of batches used during validation per candidate. If
            set to `None` use the whole validation data set.
        calib_batches (int, None):
            The number of batches used during calibration per candidate. If
            set to `None` use the whole calibration data set.
        device:
            The Pytorch device the network is moved to.

    Returns:
        list[tuple[Model, float]]:
            The final `topk` many models and their accuracy.
    """
    evo_start = time.time()

    n_cross: Final[int] = int(population_size * next_gen_split[0])
    n_mut: Final[int] = int(population_size * next_gen_split[1])

    logger.info(make_caption("Evolution", 70, " "))

    population = initial_population
    top: list[tuple[Model, float]] = []

    for i in range(iterations):
        logger.info(
            make_caption(f"Iteration {i + 1}/{iterations}", 70, "-")
        )

        iteration_start = time.time()

        top = _do_evaluate(
            supernet,
            population,
            valid_dl,
            calib_dl,
            valid_batches,
            calib_batches,
            top,
            topk,
            i,
            logger,
            device
        )
        if ctx is not None:
            ctx.iteration(i + 1, population, top)

        cross = _do_crossover(top, topk, n_cross, fitness, i, logger)
        mut = _do_mutate(
            top, topk, n_mut, mutation_rate, mutator, fitness, i, logger
        )

        population = cross + mut

        iteration_time = time.time() - iteration_start
        logger.info(f"\ntime={iteration_time:.2f}s")

    logger.info(make_caption("Final Population", 70, "-"))

    top = _do_evaluate(
            supernet,
            population,
            valid_dl,
            calib_dl,
            valid_batches,
            calib_batches,
            top,
            topk,
            -2,
            logger,
            device
        )
    if ctx is not None:
        ctx.iteration(-1, population, top)

    evo_time = time.time() - evo_start
    logger.info(f"total={evo_time:.2f}s")

    return top