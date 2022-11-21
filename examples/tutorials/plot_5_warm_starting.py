"""

*******************************
Warm Starting of HPO Algorithms
*******************************

This tutorial shows how to leverage the results of previous experiments to more efficiently search
the hyper-parameter space of a new experiment.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing_extensions import Literal

from orion.client import build_experiment
from orion.executor.single_backend import SingleExecutor

# flake8: noqa: E266

#%%
#
# Training code
#

DatasetName = Literal["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "SVHN"]

normalization_means: dict[DatasetName, tuple[float, ...]] = {
    "MNIST": (0.1307,),
    "FashionMNIST": (0.2860,),
    "CIFAR10": (0.4914, 0.4822, 0.4465),
    "CIFAR100": (0.5071, 0.4865, 0.4409),
}
normalization_stds: dict[DatasetName, tuple[float, ...]] = {
    "MNIST": (0.3081,),
    "FashionMNIST": (0.3530,),
    "CIFAR10": (0.2470, 0.2435, 0.2616),
    "CIFAR100": (0.2673, 0.2564, 0.2762),
}
dataset_num_classes: dict[DatasetName, int] = {
    "MNIST": 10,
    "FashionMNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
}


@dataclass
class Args:
    """Configuration options for a training run."""

    dataset: DatasetName = "MNIST"
    """ Dataset to use."""

    batch_size: int = 512
    """input batch size for training"""

    test_batch_size: int = 1000
    """input batch size for testing"""

    epochs: int = 13
    """number of epochs to train"""

    lr: float = 1e-2
    """learning rate"""

    gamma: float = 0.7
    """Learning rate step gamma"""

    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore
            else "cpu"
        )
    )
    """Device to use for training."""

    dry_run: bool = False
    """quickly check a single pass"""

    seed: int = 1
    """random seed (default: 1)"""

    log_interval: int = 10
    """how many batches to wait before logging training status"""

    save_model: bool = False
    """For Saving the current Model"""

    data_dir: Path = Path(os.environ.get("DATA_DIR", "data"))
    """ Directory where the dataset should be found or downloaded."""


class Net(nn.Sequential):
    """Simple convnet."""

    def __init__(self, n_classes: int = 10):
        super().__init__(
            # NOTE: `in_channels` is determined in the first forward pass
            nn.LazyConv2d(32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            # NOTE: `in_features` is determined in the first forward pass
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )


def train_epoch(
    args: Args,
    model: Net,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
):
    """Trains a model given the configuration."""
    model.train()
    loss_function = F.cross_entropy

    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({"loss": loss.item()})
            if args.dry_run:
                break


def test_epoch(model: Net, device: torch.device, test_loader: DataLoader) -> float:
    """Tests the model, returning the average test loss."""
    model.eval()
    test_loss = 0.0
    correct = 0

    num_batches = len(test_loader.dataset)  # type: ignore

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_batches

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_batches} "
        f"({100.0 * correct / num_batches:.0f}%)\n"
    )
    return test_loss


def main(**kwargs):
    """Main loop. Trains and then tests a model after each epoch."""
    # Training settings

    # note: could also use simple-parsing to parse the Config from the command-line:
    # import simple_parsing
    # from simple_parsing import parse
    # config = parse(Config)
    args = Args(**kwargs)
    print(f"Args: {args}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if device.type == "cuda":
        # Note: When using Orion with parallel workers (which is the case by default?),
        # `num_workers` should be set to 0, because otherwise we get an error about daemonic
        # processes having children, etc.
        cuda_kwargs = {"num_workers": 0, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                normalization_means[args.dataset], normalization_stds[args.dataset]
            ),
        ]
    )

    data_dir = args.data_dir
    dataset_class = getattr(datasets, args.dataset)
    train_dataset = dataset_class(
        str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(str(data_dir), train=False, transform=transform)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    model = Net(n_classes=dataset_num_classes[args.dataset]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_loss = None
    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, optimizer, epoch)
        test_loss = test_epoch(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        run_working_dir = Path(os.environ.get("ORION_WORKING_DIR", "."))
        # use the trial working dir to save the model.
        torch.save(model.state_dict(), str(run_working_dir / "model.pt"))
    return [dict(name="loss", type="objective", value=test_loss)]


# %%
# Controls for this example:
#
previous_experiment_n_runs = 10
previous_experiment_settings = {
    "dataset": "CIFAR100",
    "epochs": 3,
}

current_experiment_n_runs = 10
current_experiment_settings = {
    "dataset": "CIFAR10",
    "epochs": 3,
}

# We're using multiple seeds for a more robust comparison of with/without warm-starting.
n_seeds = 3

# The number of initial random suggestions that the optimization algorithm should do.
n_initial_random_suggestions = 5

# NOTE: This gets run in the tutorials directory
# NOTE: This needs to be a relative path, otherwise the CI runs will fail.
# Specify the database where the previous experiments are stored. We use a local PickleDB here.
previous_experiment_storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "previous_db.pkl",
    },
}
current_experiment_storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "current_db.pkl",
    },
}


previous_experiment = build_experiment(
    name="previous_experiment",
    space={"lr": "loguniform(1e-5, 1.0)"},
    storage=previous_experiment_storage,
    algorithms={"random": {"seed": 1}},
    max_trials=previous_experiment_n_runs,
    executor=SingleExecutor(),
)

# %%
# Populate the initial experiment with some trials:

previous_experiment.workon(main, **previous_experiment_settings)

# %%
# Run a new experiment, without warm-starting (a.k.a. "cold-start"):


cold_experiments = [
    build_experiment(
        name=f"cold_experiment_{seed}",
        space={"lr": "loguniform(1e-5, 1.0)"},
        storage=current_experiment_storage,
        executor=SingleExecutor(),
        algorithms={
            "tpe": {"seed": seed, "n_initial_points": n_initial_random_suggestions}
        },
        # algorithms={"robo_gp": {"seed": seed, "n_initial_points": n_initial_points}},
        max_trials=current_experiment_n_runs,
    )
    for seed in range(n_seeds)
]
for exp in cold_experiments:
    exp.workon(main, **previous_experiment_settings)

#%%
# New experiment with warm-starting:

assert previous_experiment.storage
assert previous_experiment.max_trials
warm_experiments = [
    build_experiment(
        name=f"warm_experiment_{seed}",
        space={"lr": "loguniform(1e-5, 1.0)"},
        storage=current_experiment_storage,
        executor=SingleExecutor(),
        max_trials=current_experiment_n_runs,
        # NOTE: This n_initial_points is changed slightly, since it also counts the trials from
        # the previous experiment. This is just so the comparison is a bit fairer.
        # Both algorithms do random search for the first few trials of the current task and then
        # optimize.
        algorithms={
            "tpe": {
                "seed": seed,
                "n_initial_points": previous_experiment_n_runs
                + n_initial_random_suggestions,
            }
        },
        # Pass the knowledge base to `build_experiment`, either with a configuration dictionary:
        knowledge_base={"KnowledgeBase": {"storage": previous_experiment_storage}},
        # Or by instianting a KnowledgeBase and passing it directly:
        # knowledge_base=KnowledgeBase(storage=previous_experiment.storage),
    )
    for seed in range(n_seeds)
]

for exp in warm_experiments:
    exp.workon(main, **current_experiment_settings)

# %%
#
# Compare the results:
#
# Here we use the :func:`orion.plotting.base.regrets` function to plot the results.
# This shows the performance of each variant.
# in blue, we have the single line which shows the results we had on the previous experiment.
# In yellow, we have the results of the new experiment (same model on the new dataset).
# In red, we have the warm-started experiment, where we give the algorithm access to the old data.
# The previous data gets annotated with a different task id when viewed by the algorithm, making it
# possible for the algorithm to learn what is common and what is different about each task.

from orion.plotting.base import regrets

fig = regrets(
    {
        "previous experiment": [previous_experiment],
        "without warm-start": cold_experiments,
        "with warm-start": warm_experiments,
    },
)
fig.show()
fig.write_image("../../docs/src/_static/warm_start_thumbnail.png")
fig

# sphinx_gallery_thumbnail_path = '_static/warm_start_thumbnail.png'
