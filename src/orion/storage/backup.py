# pylint: disable=,protected-access,too-many-locals,too-many-branches,too-many-statements
"""
Module responsible for storage export/import
============================================

Provide functions to export and import database content.
"""
import logging
import os
import shutil
from typing import Any, Dict, List

from orion.core.io.database import DatabaseError
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils import generate_temporary_file
from orion.core.utils.tree import TreeNode
from orion.storage.base import BaseStorageProtocol, setup_storage

logger = logging.getLogger(__name__)

COL_EXPERIMENTS = "experiments"
COL_ALGOS = "algo"
COL_BENCHMARKS = "benchmarks"
COL_TRIALS = "trials"

COLLECTIONS = {"experiments", "algo", "benchmarks", "trials"}
EXPERIMENT_RELATED_COLLECTIONS = {"algo", "trials"}

STEP_COLLECT_EXPERIMENTS = 0
STEP_CHECK_BENCHMARKS = 1
STEP_CHECK_DST_EXPERIMENTS = 2
STEP_CHECK_SRC_EXPERIMENTS = 3
STEP_DELETE_OLD_DATA = 4
STEP_INSERT_NEW_DATA = 5
STEP_NAMES = [
    "Collect source experiments to load",
    "Check benchmarks",
    "Check destination experiments",
    "Check source experiments",
    "Delete data to replace in destination",
    "Insert new data in destination",
]


def dump_database(storage, dump_host, name=None, version=None, overwrite=False):
    """Dump a database

    Parameters
    ----------
    storage: BaseStorageProtocol
        storage of database to dump
    dump_host:
        file path to dump into (dumped file will be a pickled file)
    name:
        (optional) name of experiment to dump (by default, full database is dumped)
    version:
        (optional) version of experiment to dump.
        By default, use the latest version of provided `name`.
    overwrite:
        (optional) define how to manage destination file if already exists.
        If false (default), raise an exception.
        If true, delete existing file and create a new one with dumped data.
    """
    dump_host = os.path.abspath(dump_host)

    # For pickled databases, make sure src is not dst
    if hasattr(storage, "_db"):
        orig_db = storage._db
        if isinstance(orig_db, PickledDB) and dump_host == os.path.abspath(
            orig_db.host
        ):
            raise DatabaseError("Cannot dump pickleddb to itself.")

    # Temporary output file to be used for dumping. Default is dump_host
    tmp_dump_host = dump_host

    if os.path.exists(dump_host):
        if overwrite:
            # Work on a temporary file, not directly into dump_host.
            # dump_host will then be replaced with temporary file
            # if no error occurred.
            tmp_dump_host = generate_temporary_file()
            assert os.path.exists(tmp_dump_host)
            assert os.stat(tmp_dump_host).st_size == 0
            logger.info(f"Overwriting previous output at {dump_host}")
        else:
            raise DatabaseError(
                f"Export output already exists (specify `--force` to overwrite) at {dump_host}"
            )

    try:
        dst_storage = setup_storage(
            {"database": {"host": tmp_dump_host, "type": "pickleddb"}}
        )
        logger.info(f"Dump to {dump_host}")
        _dump(storage, dst_storage, name, version)
    except Exception as exc:
        # An exception occurred when dumping.
        # If existed, original dump_host has not been modified.
        for path in (tmp_dump_host, f"{tmp_dump_host}.lock"):
            if os.path.isfile(path):
                os.unlink(path)
        raise exc
    else:
        # No error occurred
        # Move tmp_dump_host to dump_host if necessary
        if tmp_dump_host != dump_host:
            # NB: If an OS error occurs here, we can't do anything.
            os.unlink(dump_host)
            shutil.move(tmp_dump_host, dump_host)
            # Cleanup
            tmp_lock_host = f"{tmp_dump_host}.lock"
            if os.path.isfile(tmp_lock_host):
                os.unlink(tmp_lock_host)


def load_database(
    storage, load_host, resolve=None, name=None, version=None, progress_callback=None
):
    """Import data into a database

    Parameters
    ----------
    storage: BaseStorageProtocol
        storage of destination database to load into
    load_host:
        file path containing data to import
        (should be a pickled file representing a PickledDB)
    resolve:
        policy to resolve import conflict. Either None, 'ignore', 'overwrite' or 'bump'.
        - None will raise an exception on any conflict detected
        - 'ignore' will ignore imported data on conflict
        - 'overwrite' will overwrite old data in destination database on conflict
        - 'bump' will bump imported data version before adding it,
          if data with same ID is found in destination
    name:
        (optional) name of experiment to import (by default, whole file is imported)
    version:
        (optional) version of experiment to import.
        By default, use the latest version of provided `name`.
    progress_callback:
        Optional callback to report progression. Receives 2 parameters:
        - step description (string)
        - overall progress (0 <= floating value <= 1)
    """
    load_host = os.path.abspath(load_host)

    # For pickled databases, make sure src is not dst
    if hasattr(storage, "_db"):
        dst_db = storage._db
        if isinstance(dst_db, PickledDB) and load_host == os.path.abspath(dst_db.host):
            raise DatabaseError("Cannot load pickleddb to itself.")

    src_storage: BaseStorageProtocol = setup_storage(
        {"database": {"host": load_host, "type": "pickleddb"}}
    )
    logger.info(f"Loaded src {load_host}")

    import_benchmarks = False
    _describe_import_progress(STEP_COLLECT_EXPERIMENTS, 0, 1, progress_callback)
    if name is None:
        import_benchmarks = True
        # Retrieve all src experiments for export
        experiments = src_storage.fetch_experiments({})
    else:
        # Find experiments based on given name and version
        query = {"name": name}
        if version is not None:
            query["version"] = version
        experiments = src_storage.fetch_experiments(query)
        if not experiments:
            raise DatabaseError(
                f"No experiment found with query {query}. Nothing to import."
            )
        if len(experiments) > 1:
            experiments = sorted(experiments, key=lambda d: d["version"])[-1:]
        logger.info(
            f"Found experiment {experiments[0]['name']}.{experiments[0]['version']}"
        )
    _describe_import_progress(STEP_COLLECT_EXPERIMENTS, 1, 1, progress_callback)

    preparation = _prepare_import(
        src_storage,
        storage,
        experiments,
        resolve,
        import_benchmarks,
        progress_callback=progress_callback,
    )
    _execute_import(storage, *preparation, progress_callback=progress_callback)


def _dump(src_storage, dst_storage, name=None, version=None):
    """Dump data from source storage to destination storage.

    Parameters
    ----------
    src_storage: BaseStorageProtocol
        input storage
    dst_storage: BaseStorageProtocol
        output storage
    name:
        (optional) if provided, dump only data related to experiment with this name
    version:
        (optional) version of experiment to dump
    """
    # Get collection names in a set
    if name is None:
        # Nothing to filter, dump everything
        # Dump benchmarks
        logger.info("Dumping benchmarks")
        for benchmark in src_storage.fetch_benchmark({}):
            dst_storage.create_benchmark(benchmark)
        # Dump experiments
        logger.info("Dumping experiments, algos and trials")
        # Dump experiments ordered from parents to children,
        # so that we can get new parent IDs from dst
        # before writing children.
        graph = get_experiment_parent_links(src_storage.fetch_experiments({}))
        sorted_experiments = graph.get_sorted_data()
        src_to_dst_id = {}
        for i, src_exp in enumerate(sorted_experiments):
            logger.info(
                f"Dumping experiment {i + 1}: {src_exp['name']}.{src_exp['version']}"
            )
            _dump_experiment(src_storage, dst_storage, src_exp, src_to_dst_id)
    else:
        # Get experiments with given name
        query = {"name": name}
        if version is not None:
            query["version"] = version
        experiments = src_storage.fetch_experiments(query)
        if not experiments:
            raise DatabaseError(
                f"No experiment found with query {query}. Nothing to dump."
            )
        exp_data = sorted(experiments, key=lambda d: d["version"])[-1]
        logger.info(f"Found experiment {exp_data['name']}.{exp_data['version']}")
        # As we dump only 1 experiment, remove parent links if exist
        if exp_data["refers"]:
            if exp_data["refers"]["root_id"] is not None:
                logger.info("Removing reference to root experiment before dumping")
                exp_data["refers"]["root_id"] = None
            if exp_data["refers"]["parent_id"] is not None:
                logger.info("Removing reference to parent experiment before dumping")
                exp_data["refers"]["parent_id"] = None
        # Dump selected experiments and related data
        logger.info(f"Dumping experiment {name}")
        _dump_experiment(src_storage, dst_storage, exp_data, {})


def _dump_experiment(src_storage, dst_storage, src_exp, src_to_dst_id: dict):
    """Dump a single experiment and related data from src to dst storage.

    Parameters
    ----------
    src_storage:
        src storage
    dst_storage:
        dst storage
    src_exp: dict
        src experiment
    src_to_dst_id: dict
        Dictionary mapping experiment ID from src to dst.
        Used to set dst parent ID when writing child experiment in dst storage.
        Updated with new dst ID corresponding to `src_exp`.
    """
    _write_experiment(
        src_exp,
        algo_lock_info=src_storage.get_algorithm_lock_info(uid=src_exp["_id"]),
        trials=src_storage.fetch_trials(uid=src_exp["_id"]),
        dst_storage=dst_storage,
        src_to_dst_id=src_to_dst_id,
        verbose=True,
    )


def _prepare_import(
    src_storage,
    dst_storage,
    experiments,
    resolve=None,
    import_benchmarks=True,
    progress_callback=None,
):
    """Prepare importation.

    Compute all changes to apply to make import and return changes as dictionaries.

    Parameters
    ----------
    src_storage: BaseStorageProtocol
        storage to import from
    dst_storage: BaseStorageProtocol
        storage to import into
    experiments:
        experiments to import from src_storage into dst_storage
    resolve:
        resolve policy
    import_benchmarks:
        if True, benchmarks will be also imported from src_database
    progress_callback:
        See :func:`load_database`

    Returns
    -------
    A couple (queries to delete, data to add) representing
    changes to apply to dst_storage to make import
    """
    assert resolve is None or resolve in ("ignore", "overwrite", "bump")

    queries_to_delete = {}
    data_to_add = {}

    if import_benchmarks:
        src_benchmarks = src_storage.fetch_benchmark({})
        for i, src_benchmark in enumerate(src_benchmarks):
            _describe_import_progress(
                STEP_CHECK_BENCHMARKS, i, len(src_benchmarks), progress_callback
            )
            dst_benchmarks = dst_storage.fetch_benchmark(
                {"name": src_benchmark["name"]}
            )
            if dst_benchmarks:
                (dst_benchmark,) = dst_benchmarks
                if resolve == "ignore":
                    logger.info(
                        f'Ignored benchmark already in dst: {src_benchmark["name"]}'
                    )
                    continue
                if resolve == "overwrite":
                    logger.info(
                        f'Overwrite benchmark in dst, name: {src_benchmark["name"]}'
                    )
                    queries_to_delete.setdefault(COL_BENCHMARKS, []).append(
                        {"_id": dst_benchmark["_id"]}
                    )
                elif resolve == "bump":
                    raise DatabaseError(
                        "Can't bump benchmark version, "
                        "as benchmarks do not currently support versioning."
                    )
                else:  # resolve is None or unknown
                    raise DatabaseError(
                        f"Conflict detected without strategy to resolve ({resolve}) "
                        f"for benchmark {src_benchmark['name']}"
                    )
            # Delete benchmark database ID so that a new one will be generated on insertion
            del src_benchmark["_id"]
            data_to_add.setdefault(COL_BENCHMARKS, []).append(src_benchmark)
        _describe_import_progress(
            STEP_CHECK_BENCHMARKS,
            len(src_benchmarks),
            len(src_benchmarks),
            progress_callback,
        )

    _describe_import_progress(STEP_CHECK_DST_EXPERIMENTS, 0, 1, progress_callback)
    all_dst_experiments = dst_storage.fetch_experiments({})
    # Dictionary mapping dst exp name to exp version to list of exps with same name and version
    dst_exp_map = {}
    last_versions = {}
    for dst_exp in all_dst_experiments:
        name = dst_exp["name"]
        version = dst_exp["version"]
        last_versions[name] = max(last_versions.get(name, 0), version)
        dst_exp_map.setdefault(name, {}).setdefault(version, []).append(dst_exp)
    _describe_import_progress(STEP_CHECK_DST_EXPERIMENTS, 1, 1, progress_callback)

    if len(experiments) == 1:
        # As we load only 1 experiment, remove parent links if exist
        (exp_data,) = experiments
        if exp_data["refers"]:
            if exp_data["refers"]["root_id"] is not None:
                logger.info("Removing reference to root experiment before loading")
                exp_data["refers"]["root_id"] = None
            if exp_data["refers"]["parent_id"] is not None:
                logger.info("Removing reference to parent experiment before loading")
                exp_data["refers"]["parent_id"] = None
    else:
        # Load experiments ordered from parents to children,
        # so that we can get new parent IDs from dst
        # before writing children.
        graph = get_experiment_parent_links(experiments)
        experiments = graph.get_sorted_data()

    for i, experiment in enumerate(experiments):
        _describe_import_progress(
            STEP_CHECK_SRC_EXPERIMENTS, i, len(experiments), progress_callback
        )
        dst_experiments = dst_exp_map.get(experiment["name"], {}).get(
            experiment["version"], []
        )
        if dst_experiments:
            (dst_experiment,) = dst_experiments
            if resolve == "ignore":
                logger.info(
                    f"Ignored experiment already in dst: "
                    f'{experiment["name"]}.{experiment["version"]}'
                )
                continue
            if resolve == "overwrite":
                # We must remove experiment data in dst
                logger.info(
                    f"Overwrite experiment in dst: "
                    f'{dst_experiment["name"]}.{dst_experiment["version"]}'
                )
                for collection in EXPERIMENT_RELATED_COLLECTIONS:
                    queries_to_delete.setdefault(collection, []).append(
                        {"experiment": dst_experiment["_id"]}
                    )
                queries_to_delete.setdefault(COL_EXPERIMENTS, []).append(
                    {"_id": dst_experiment["_id"]}
                )
            elif resolve == "bump":
                old_version = experiment["version"]
                new_version = last_versions[experiment["name"]] + 1
                last_versions[experiment["name"]] = new_version
                experiment["version"] = new_version
                logger.info(
                    f'Bumped version of src experiment: {experiment["name"]}, '
                    f"from {old_version} to {new_version}"
                )
            else:  # resolve is None or unknown
                raise DatabaseError(
                    f"Conflict detected without strategy to resolve ({resolve}) "
                    f"for experiment {experiment['name']}.{experiment['version']}"
                )
        else:
            logger.info(
                f'Import experiment {experiment["name"]}.{experiment["version"]}'
            )

        # Get data related to experiment to import.
        algo = src_storage.get_algorithm_lock_info(uid=experiment["_id"])
        trials = src_storage.fetch_trials(uid=experiment["_id"])
        # We will use experiment key to link experiment to related data.
        exp_key = _get_exp_key(experiment)
        # Set data to add
        data_to_add.setdefault(COL_EXPERIMENTS, []).append(experiment)
        data_to_add.setdefault(COL_ALGOS, {})[exp_key] = algo
        data_to_add.setdefault(COL_TRIALS, {})[exp_key] = trials
    _describe_import_progress(
        STEP_CHECK_SRC_EXPERIMENTS,
        len(experiments),
        len(experiments),
        progress_callback,
    )

    return queries_to_delete, data_to_add


def _execute_import(
    dst_storage, queries_to_delete, data_to_add, progress_callback=None
):
    """Execute import

    Parameters
    ----------
    dst_storage: BaseStorageProtocol
        destination storage where to apply changes
    queries_to_delete: dict
        dictionary mapping a collection name to a list of queries to use
        to find and delete data
    data_to_add: dict
        dictionary mapping a collection name to a list of data to add
    progress_callback:
        See :func:`load_database`
    """

    # Delete data

    total_queries = sum(len(queries) for queries in queries_to_delete.values())
    for collection_name in COLLECTIONS:
        queries_to_delete.setdefault(collection_name, ())
    i_query = 0
    for query_delete_benchmark in queries_to_delete[COL_BENCHMARKS]:
        logger.info(
            f"Deleting from {len(queries_to_delete[COL_BENCHMARKS])} queries into {COL_BENCHMARKS}"
        )
        dst_storage.delete_benchmark(query_delete_benchmark)
        _describe_import_progress(
            STEP_DELETE_OLD_DATA, i_query, total_queries, progress_callback
        )
        i_query += 1
    for query_delete_experiment in queries_to_delete[COL_EXPERIMENTS]:
        logger.info(
            f"Deleting from {len(queries_to_delete[COL_EXPERIMENTS])} queries "
            f"into {COL_EXPERIMENTS}"
        )
        dst_storage.delete_experiment(uid=query_delete_experiment["_id"])
        _describe_import_progress(
            STEP_DELETE_OLD_DATA, i_query, total_queries, progress_callback
        )
        i_query += 1
    for query_delete_trials in queries_to_delete[COL_TRIALS]:
        logger.info(
            f"Deleting from {len(queries_to_delete[COL_TRIALS])} queries into {COL_TRIALS}"
        )
        dst_storage.delete_trials(uid=query_delete_trials["experiment"])
        _describe_import_progress(
            STEP_DELETE_OLD_DATA, i_query, total_queries, progress_callback
        )
        i_query += 1
    for query_delete_algo in queries_to_delete[COL_ALGOS]:
        logger.info(
            f"Deleting from {len(queries_to_delete[COL_ALGOS])} queries into {COL_ALGOS}"
        )
        dst_storage.delete_algorithm_lock(uid=query_delete_algo["experiment"])
        _describe_import_progress(
            STEP_DELETE_OLD_DATA, i_query, total_queries, progress_callback
        )
        i_query += 1

    _describe_import_progress(
        STEP_DELETE_OLD_DATA, total_queries, total_queries, progress_callback
    )

    # Add data

    nb_data_to_add = len(data_to_add.get(COL_BENCHMARKS, ())) + len(
        data_to_add.get(COL_EXPERIMENTS, ())
    )
    i_data = 0

    for new_benchmark in data_to_add.get(COL_BENCHMARKS, ()):
        dst_storage.create_benchmark(new_benchmark)
        _describe_import_progress(
            STEP_INSERT_NEW_DATA, i_data, nb_data_to_add, progress_callback
        )
        i_data += 1

    src_to_dst_id = {}
    for src_exp in data_to_add.get(COL_EXPERIMENTS, ()):
        exp_key = _get_exp_key(src_exp)
        new_algo = data_to_add[COL_ALGOS][exp_key]
        new_trials = data_to_add[COL_TRIALS][exp_key]
        _write_experiment(
            src_exp,
            algo_lock_info=new_algo,
            trials=new_trials,
            dst_storage=dst_storage,
            src_to_dst_id=src_to_dst_id,
            verbose=False,
        )
        _describe_import_progress(
            STEP_INSERT_NEW_DATA, i_data, nb_data_to_add, progress_callback
        )
        i_data += 1

    _describe_import_progress(
        STEP_INSERT_NEW_DATA, nb_data_to_add, nb_data_to_add, progress_callback
    )


def _write_experiment(
    src_exp, algo_lock_info, trials, dst_storage, src_to_dst_id: dict, verbose=False
):
    # Remove src experiment database ID
    src_id = src_exp.pop("_id")
    assert src_id not in src_to_dst_id

    # Update experiment parent ID
    old_parent_id = _get_exp_parent_id(src_exp)
    if old_parent_id is not None:
        _set_exp_parent_id(src_exp, src_to_dst_id[old_parent_id])

    # Update experiment root ID if different from experiment ID
    old_root_id = _get_exp_root_id(src_exp)
    if old_root_id is not None:
        if old_root_id != src_id:
            _set_exp_root_id(src_exp, src_to_dst_id[old_root_id])

    # Dump experiment and algo
    dst_storage.create_experiment(
        src_exp,
        algo_locked=algo_lock_info.locked,
        algo_state=algo_lock_info.state,
        algo_heartbeat=algo_lock_info.heartbeat,
    )
    if verbose:
        logger.info("\tCreated exp")
    # Link experiment src ID to dst ID
    (dst_exp,) = dst_storage.fetch_experiments(
        {"name": src_exp["name"], "version": src_exp["version"]}
    )
    src_to_dst_id[src_id] = dst_exp["_id"]
    # Update root ID if equals to experiment ID
    if old_root_id is not None and old_root_id == src_id:
        _set_exp_root_id(src_exp, src_to_dst_id[src_id])
        dst_storage.update_experiment(
            uid=src_to_dst_id[src_id], refers=src_exp["refers"]
        )
    # Dump trials
    trial_old_to_new_id = {}
    for trial in trials:
        old_id = trial.id
        # Set trial parent to new dst exp ID
        trial.experiment = src_to_dst_id[trial.experiment]
        # Remove src trial database ID, so that new ID will be generated at insertion.
        # Trial parents are identified using trial identifier (trial.id)
        # which is not related to trial database ID (trial.id_override).
        # So, we can safely remove trial database ID.
        trial.id_override = None
        if trial.parent is not None:
            trial.parent = trial_old_to_new_id[trial.parent]
        dst_trial = dst_storage.register_trial(trial)
        trial_old_to_new_id[old_id] = dst_trial.id
    if verbose:
        logger.info("\tDumped trials")


def _describe_import_progress(step, value, total, callback=None):
    print("STEP", step + 1, STEP_NAMES[step], value, total)
    if callback:
        if total == 0:
            value = total = 1
        callback(STEP_NAMES[step], (step + (value / total)) / len(STEP_NAMES))


class _Graph:
    """Helper class to build experiments or trials graph."""

    def __init__(self, key_to_data: dict):
        """Initialize

        Parameters
        ----------
        key_to_data:
            Dictionary mapping key (used as node) to related object.
        """
        self.key_to_data: dict = key_to_data
        self.key_to_node: Dict[Any, TreeNode] = {}
        self.root = TreeNode(None)

    def add_link(self, parent, child):
        """Link parent node to child node."""
        if parent not in self.key_to_node:
            self.key_to_node[parent] = TreeNode(parent, self.root)
        if child in self.key_to_node:
            child_node = self.key_to_node[child]
            # A node should have at most 1 parent.
            assert child_node.parent is self.root
            child_node.set_parent(self.key_to_node[parent])
        else:
            self.key_to_node[child] = TreeNode(child, self.key_to_node[parent])

    def _get_sorted_nodes(self) -> List[TreeNode]:
        """Return list of sorted nodes from parents to children."""
        # Exclude root node.
        return list(self.root)[1:]

    def get_sorted_data(self) -> list:
        """Return list of sorted data from parents to children."""
        return [self.key_to_data[node.item] for node in self._get_sorted_nodes()]

    def get_sorted_links(self):
        """Return sorted edges (node, child)"""
        for node in self._get_sorted_nodes():
            if node.children:
                for child_node in node.children:
                    yield node.item, child_node.item
            else:
                yield node.item, None


def get_experiment_parent_links(experiments: list) -> _Graph:
    """Generate experiments graphs based on experiment parents.

    Does not currently check experiment roots.
    """
    graph = _Graph({_get_exp_key(exp): exp for exp in experiments})
    exp_id_to_key = {exp["_id"]: _get_exp_key(exp) for exp in experiments}
    for exp in experiments:
        parent_id = _get_exp_parent_id(exp)
        if parent_id is not None:
            parent_key = exp_id_to_key[parent_id]
            child_key = _get_exp_key(exp)
            graph.add_link(parent_key, child_key)
    return graph


def get_experiment_root_links(experiments: list) -> _Graph:
    """Generate experiments graphs based on experiment roots."""
    special_root_key = ("__root__",)
    graph = _Graph(
        {**{_get_exp_key(exp): exp for exp in experiments}, **{special_root_key: None}}
    )
    exp_id_to_key = {exp["_id"]: _get_exp_key(exp) for exp in experiments}
    for exp in experiments:
        root_id = _get_exp_root_id(exp)
        if root_id is not None:
            if root_id == exp["_id"]:
                # If root is exp, use a special root key
                root_key = special_root_key
            else:
                root_key = exp_id_to_key[root_id]
            child_key = _get_exp_key(exp)
            graph.add_link(root_key, child_key)
    return graph


def get_trial_parent_links(trials: list) -> _Graph:
    """Generate trials graph based on trial parents. Not yet used."""
    trial_map = {_get_trial_key(trial): trial for trial in trials}
    graph = _Graph(trial_map)
    for trial in trials:
        parent = _get_trial_parent(trial)
        if parent is not None:
            assert parent in trial_map
            graph.add_link(parent, _get_trial_key(trial))
    return graph


def _get_trial_key(trial):
    """Return trial key, as trial ID"""
    return trial["id"] if isinstance(trial, dict) else trial.id


def _get_trial_parent(trial):
    """Return trial parent"""
    return trial["parent"] if isinstance(trial, dict) else trial.parent


def _get_exp_key(exp: dict) -> tuple:
    """Return experiment key as tuple (name, version)"""
    return exp["name"], exp["version"]


def _get_exp_parent_id(exp: dict):
    """Get experiment parent ID or None if unavailable"""
    return exp.get("refers", {}).get("parent_id", None)


def _set_exp_parent_id(exp: dict, parent_id):
    """Set experiment parent ID"""
    exp.setdefault("refers", {})["parent_id"] = parent_id


def _get_exp_root_id(exp: dict):
    """Get experiment root ID or None if unavailable"""
    return exp.get("refers", {}).get("root_id", None)


def _set_exp_root_id(exp: dict, parent_id):
    """Set experiment root ID"""
    exp.setdefault("refers", {})["root_id"] = parent_id
