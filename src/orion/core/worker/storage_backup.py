# pylint: disable=,protected-access,too-many-locals,too-many-branches,too-many-statements
"""
Module responsible for storage export/import
============================================

Provide functions to export and import database content.
"""
import logging
import os

from orion.core.io.database import DatabaseError
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import BaseStorageProtocol, setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


def dump_database(storage, dump_host, name=None, version=None):
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
        (optional) version of experiment to dump
    """
    dump_host = os.path.abspath(dump_host)

    # For pickled databases, make sure src is not dst
    if hasattr(storage, "_db"):
        orig_db = storage._db
        if isinstance(orig_db, PickledDB) and dump_host == os.path.abspath(
            orig_db.host
        ):
            raise DatabaseError("Cannot dump pickleddb to itself.")

    dst_storage = setup_storage({"database": {"host": dump_host, "type": "pickleddb"}})
    logger.info(f"Dump to {dump_host}")
    _dump(storage, dst_storage, name, version)


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
        (optional) version of experiment to import
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
            experiments = sorted(experiments, key=lambda d: d["version"])[:1]
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
        for i, src_exp in enumerate(src_storage.fetch_experiments({})):
            logger.info(f"Dumping experiment {i + 1}")
            _dump_experiment(src_storage, dst_storage, src_exp)
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
        if len(experiments) > 1:
            exp_data = sorted(experiments, key=lambda d: d["version"])[0]
        else:
            (exp_data,) = experiments
        logger.info(f"Found experiment {exp_data['name']}.{exp_data['version']}")
        # Dump selected experiments and related data
        logger.info(f"Dumping experiment {name}")
        _dump_experiment(src_storage, dst_storage, exp_data)


def _dump_experiment(src_storage, dst_storage, src_exp):
    """Dump a single experiment and related data from src to dst storage."""
    algo_lock_info = src_storage.get_algorithm_lock_info(uid=src_exp["_id"])
    logger.info("\tGot algo lock")
    # Dump experiment and algo
    dst_storage.create_experiment(
        src_exp,
        algo_locked=algo_lock_info.locked,
        algo_state=algo_lock_info.state,
        algo_heartbeat=algo_lock_info.heartbeat,
    )
    logger.info("\tCreated exp")
    # Dump trials
    for trial in src_storage.fetch_trials(uid=src_exp["_id"]):
        dst_storage.register_trial(trial)
    logger.info("\tDumped trials")


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
    last_experiment_id = max((data["_id"] for data in all_dst_experiments), default=0)
    last_versions = {}
    for dst_exp in all_dst_experiments:
        name = dst_exp["name"]
        version = dst_exp["version"]
        last_versions[name] = max(last_versions.get(name, 0), version)
        dst_exp_map.setdefault(name, {}).setdefault(version, []).append(dst_exp)
    _describe_import_progress(STEP_CHECK_DST_EXPERIMENTS, 1, 1, progress_callback)

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
                new_version = last_versions.get(experiment["name"], 0) + 1
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

        # Generate new experiment ID
        new_experiment_id = last_experiment_id + 1
        last_experiment_id = new_experiment_id
        experiment["_id"] = new_experiment_id
        # Update trials: set new experiment ID and remove trials database IDs,
        # so that new IDs will be generated at insertion.
        # Trial parents are identified using trial identifier (trial.id)
        # which is not related to trial database ID (trial.id_override).
        # So, we can safely remove trial database ID.
        for trial in trials:
            trial.experiment = new_experiment_id
            trial.id_override = None

        # Set data to add
        data_to_add.setdefault(COL_EXPERIMENTS, []).append(experiment)
        # Link algo with new experiment ID
        data_to_add.setdefault(COL_ALGOS, {})[new_experiment_id] = algo
        data_to_add.setdefault(COL_TRIALS, {})[new_experiment_id] = trials
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

    for new_experiment in data_to_add.get(COL_EXPERIMENTS, ()):
        new_algo = data_to_add[COL_ALGOS][new_experiment["_id"]]
        new_trials = data_to_add[COL_TRIALS][new_experiment["_id"]]
        dst_storage.create_experiment(
            new_experiment,
            algo_locked=new_algo.locked,
            algo_state=new_algo.state,
            algo_heartbeat=new_algo.heartbeat,
        )
        for trial in new_trials:
            dst_storage.register_trial(trial)
        _describe_import_progress(
            STEP_INSERT_NEW_DATA, i_data, nb_data_to_add, progress_callback
        )
        i_data += 1

    _describe_import_progress(
        STEP_INSERT_NEW_DATA, nb_data_to_add, nb_data_to_add, progress_callback
    )


def _describe_import_progress(step, value, total, callback=None):
    print("STEP", step + 1, STEP_NAMES[step], value, total)
    if callback:
        if total == 0:
            value = total = 1
        callback(STEP_NAMES[step], (step + (value / total)) / len(STEP_NAMES))
