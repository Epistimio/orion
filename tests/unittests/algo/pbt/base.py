"""Example usage and tests for :mod:`orion.algo.random`."""
from orion.algo.pbt.exploit import BaseExploit
from orion.algo.pbt.explore import BaseExplore
from orion.algo.pbt.pbt import LineageNode, Lineages
from orion.core.utils.pptree import print_tree
from orion.core.worker.trial import Trial


def build_full_tree(depth, child_per_parent=2, starting_objective=1):
    """Build a full tree

    Parameters
    ----------
    depth: int
        Depth of the tree

    child_per_parent: int, optional
        Number of child per node. Default: 2
    """

    def create_node_item(node_index):
        return TrialStub(id=f"id-{node_index}", objective=node_index)

    node_index = starting_objective
    root = LineageNode(create_node_item(node_index))
    node_index += 1
    node_buffer = [root]
    next_nodes = []
    for i in range(depth - 1):
        for node in node_buffer:
            for k in range(child_per_parent):
                next_nodes.append(
                    LineageNode(create_node_item(node_index), parent=node)
                )
                node_index += 1
        node_buffer = next_nodes
        next_nodes = []

    print_tree(root, nameattr="tree_name")

    return root


def build_population(objectives):
    depth = len(objectives)
    size = len(objectives[0])
    lineages = Lineages()

    for lineage_index in range(size):
        lineages.add(
            TrialStub(
                id=f"lineage-{lineage_index}-0",
                objective=objectives[0][lineage_index],
            )
        )

    for generation in range(1, depth):
        for lineage_index in range(size):
            new_trial = TrialStub(
                id=f"lineage-{lineage_index}-{generation}",
                objective=objectives[generation][lineage_index],
            )
            parent_trial = TrialStub(id=f"lineage-{lineage_index}-{generation-1}")
            if lineage_index == ((generation - 1) % size):
                next_index = (lineage_index + 1) % len(lineages)
                base_trial = parent_trial
                parent_trial = TrialStub(id=f"lineage-{next_index}-{generation-1}")
                lineages.fork(parent_trial, new_trial)
                lineages.set_jump(base_trial, new_trial)
            else:
                lineages.fork(parent_trial, new_trial)

    return lineages


def compare_generations(trials, population_size, depth):
    trial_ids = {trial.id for trial in trials}
    expected_ids = {f"lineage-{i}-{depth}" for i in range(population_size)}
    assert trial_ids == expected_ids


class RNGStub:
    pass


def sample_trials(
    space,
    num,
    seed=1,
    status=None,
    objective=None,
    params=None,
    exp_working_dir="/nothing",
):
    if params is None:
        params = {"f": space["f"].original_dimension.original_dimension.low}

    trials = space.sample(num, seed=seed)
    new_trials = []
    for trial in trials:
        if params:
            trial = trial.branch(params=params)

        trial = space.transform(space.reverse(trial))
        trial.experiment = 1

        trial.exp_working_dir = exp_working_dir

        if status:
            trial.status = status
        if status == "completed" and objective is not None:
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=1)
            )

        new_trials.append(trial)

    return new_trials


def build_lineages_for_exploit(
    space, monkeypatch, trials=None, elites=None, additional_trials=None, seed=1, num=10
):
    if trials is None:
        trials = space.sample(num, seed=seed)
        for i, trial in enumerate(trials):
            trial.status = "completed"
            trial._results.append(
                trial.Result(name="objective", type="objective", value=i)
            )
    if elites is None:
        elites = space.sample(num, seed=seed + 1)
        for i, trial in enumerate(elites):
            trial.status = "completed"
            trial._results.append(
                trial.Result(name="objective", type="objective", value=i * 2)
            )

    if additional_trials:
        trials += additional_trials

    def return_trials(*args, **kwargs):
        return trials

    def return_elites(*args, **kwargs):
        return elites

    lineages = Lineages()
    monkeypatch.setattr(lineages, "get_trials_at_depth", return_trials)
    monkeypatch.setattr(lineages, "get_elites", return_elites)

    return lineages


class ObjectiveStub:
    def __init__(self, value):
        self.value = value


class TrialStub:
    def __init__(
        self,
        working_dir=None,
        objective=None,
        id=None,
        status=None,
        params=None,
        parent=None,
    ):
        self.id = id
        if working_dir is None:
            working_dir = id

        self.working_dir = working_dir
        if objective and (status is None or status == "completed"):
            self.objective = ObjectiveStub(objective)
        else:
            self.objective = None

        if status is None and objective is not None:
            self.status = "completed"
        elif status is None:
            self.status = "new"
        else:
            self.status = status

        self.params = params
        self.parent = parent

    def __repr__(self):
        return self.id


class ExploitStub(BaseExploit):
    def __init__(self, rval=None, skip=False, should_receive=None, **kwargs):
        self.rval = rval
        self.skip = skip
        self.should_receive = should_receive
        self.kwargs = kwargs

    def __call__(self, rng, trial, lineages):
        if self.should_receive:
            assert trial is self.should_receive

        if self.skip:
            return None

        if self.rval is not None:
            return self.rval

        return trial

    @property
    def configuration(self):
        configuration = super().configuration
        configuration["rval"] = self.rval
        configuration["skip"] = self.skip
        configuration["should_receive"] = self.should_receive
        configuration.update(self.kwargs)
        return configuration


class ExploreStub(BaseExplore):
    def __init__(self, rval=None, no_call=False, **kwargs):
        self.rval = rval
        self.no_call = no_call
        self.kwargs = kwargs

    def __call__(self, rng, space, params):
        if self.no_call:
            raise RuntimeError("Should not have been called!")

        if self.rval is not None:
            return self.rval

        return params

    @property
    def configuration(self):
        configuration = super().configuration
        configuration["rval"] = self.rval
        configuration["no_call"] = self.no_call
        configuration.update(self.kwargs)
        return configuration
