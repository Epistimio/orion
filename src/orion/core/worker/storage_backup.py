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
from orion.core.worker.trial import Trial
from orion.storage.base import setup_storage

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


def _describe_import_progress(step, value, total, callback=None):
    print("STEP", step + 1, STEP_NAMES[step], value, total)
    if callback:
        if total == 0:
            value = total = 1
        callback(STEP_NAMES[step], (step + (value / total)) / len(STEP_NAMES))


def dump_database(storage, dump_host, name=None, version=None):
    """Dump a database
    :param storage: storage of database to dump
    :param dump_host: file path to dump into (dumped file will be a pickled file)
    :param name: (optional) name of experiment to dump (by default, full database is dumped)
    :param version: (optional) version of experiment to dump
    """
    orig_db = storage._db
    dump_host = os.path.abspath(dump_host)
    if isinstance(orig_db, PickledDB) and dump_host == os.path.abspath(orig_db.host):
        raise DatabaseError("Cannot dump pickleddb to itself.")
    dst_storage = setup_storage({"database": {"host": dump_host, "type": "pickleddb"}})
    db = dst_storage._db
    logger.info(f"Dump to {db}")
    if isinstance(orig_db, PickledDB):
        with orig_db.locked_database(write=False) as database:
            _dump(database, db, COLLECTIONS, name, version)
    else:
        _dump(orig_db, db, COLLECTIONS, name, version)


def load_database(
    storage, load_host, resolve, name=None, version=None, progress_callback=None
):
    """Import data into a database
    :param storage: storage of destination database to load into
    :param load_host: file path containing data to import
        (should be a pickled file representing a PickledDB)
    :param resolve: policy to resolve import conflict. Either 'ignore', 'overwrite' or 'bump'
        'ignore' will ignore imported data on conflict
        'overwrite' will overwrite old data in destination database on conflict
        'bump' will bump imported data version before adding it,
            if data with same ID is found in destination
    :param name: (optional) name of experiment to import (by default, whole file is imported)
    :param version: (optional) version of experiment to import
    """
    src_storage = setup_storage(
        {"database": {"host": os.path.abspath(load_host), "type": "pickleddb"}}
    )
    src_db = src_storage._db
    logger.info(f"Loaded src {src_db}")

    dst_db = storage._db
    import_benchmarks = False
    with src_db.locked_database(write=False) as src_database:
        _describe_import_progress(STEP_COLLECT_EXPERIMENTS, 0, 1, progress_callback)
        if name is None:
            import_benchmarks = True
            # Retrieve all src experiments for export
            experiments = src_database.read("experiments")
        else:
            # Find experiments based on given name and version
            query = {"name": name}
            if version is not None:
                query["version"] = version
            experiments = src_database.read("experiments", query)
            if not experiments:
                raise DatabaseError(
                    f"No experiment found with query {query}. Nothing to dump."
                )
            if len(experiments) > 1:
                experiments = sorted(experiments, key=lambda d: d["version"])[:1]
            logger.info(
                f"Found experiment {experiments[0]['name']}.{experiments[0]['version']}"
            )
        _describe_import_progress(STEP_COLLECT_EXPERIMENTS, 1, 1, progress_callback)
        preparation = _prepare_import(
            src_database,
            dst_db,
            resolve,
            experiments,
            import_benchmarks,
            progress_callback=progress_callback,
        )
        _execute_import(dst_db, *preparation, progress_callback=progress_callback)


def _dump(src_db, dst_db, collection_names, name=None, version=None):
    """
    Dump data from database to db.
    :param src_db: input database
    :param dst_db: output database
    :param collection_names: set of collection names to dump
    :param name: (optional) if provided, dump only
        data related to experiment with this name
    :param version: (optional) version of experiment to dump
    """
    # Get collection names in a set
    if name is None:
        # Nothing to filter, dump everything
        for collection_name in collection_names:
            logger.info(f"Dumping collection {collection_name}")
            data = src_db.read(collection_name)
            dst_db.write(collection_name, data)
    else:
        # Get experiments with given name
        assert "experiments" in collection_names
        query = {"name": name}
        if version is not None:
            query["version"] = version
        experiments = src_db.read("experiments", query)
        if not experiments:
            raise DatabaseError(
                f"No experiment found with query {query}. Nothing to dump."
            )
        if len(experiments) > 1:
            exp_data = sorted(experiments, key=lambda d: d["version"])[0]
        else:
            (exp_data,) = experiments
        logger.info(f"Found experiment {exp_data['name']}.{exp_data['version']}")
        # Dump selected experiments
        logger.info(f"Dumping experiment {name}")
        dst_db.write("experiments", exp_data)
        # Dump data related to selected experiments (do not dump other experiments)
        for collection_name in sorted(collection_names - {"experiments"}):
            filtered_data = [
                element
                for element in src_db.read(collection_name)
                if element.get("experiment", None) == exp_data["_id"]
            ]
            dst_db.write(collection_name, filtered_data)
            logger.info(
                f"Written {len(filtered_data)} filtered data "
                f"for collection {collection_name}"
            )


def _prepare_import(
    src_database,
    dst_db,
    resolve,
    experiments,
    import_benchmarks=True,
    progress_callback=None,
):
    """Prepare importation.

    Compute all changes to apply to make import and return changes as dictionaries.

    :param src_database: database to import from
    :param dst_db: database to import into
    :param resolve: resolve policy
    :param experiments: experiments to import from src_database into dst_db
    :param import_benchmarks: if True, benchmarks will be also imported from src_database
    :return: a couple (queries to delete, data to add) representing
        changes to apply to dst_db to make import
    """

    queries_to_delete = {}
    data_to_add = {}

    if import_benchmarks:
        src_benchmarks = src_database.read(COL_BENCHMARKS)
        for i, src_benchmark in enumerate(src_benchmarks):
            _describe_import_progress(
                STEP_CHECK_BENCHMARKS, i, len(src_benchmarks), progress_callback
            )
            dst_benchmarks = dst_db.read(
                COL_BENCHMARKS, {"name": src_benchmark["name"]}
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
    all_dst_experiments = dst_db.read("experiments")
    last_experiment_id = max((data["_id"] for data in all_dst_experiments), default=0)
    last_versions = {}
    for dst_exp in all_dst_experiments:
        name = dst_exp["name"]
        version = dst_exp["version"]
        last_versions[name] = max(last_versions.get(name, 0), version)
    _describe_import_progress(STEP_CHECK_DST_EXPERIMENTS, 1, 1, progress_callback)

    for i, experiment in enumerate(experiments):
        _describe_import_progress(
            STEP_CHECK_SRC_EXPERIMENTS, i, len(experiments), progress_callback
        )
        dst_experiments = dst_db.read(
            "experiments",
            {"name": experiment["name"], "version": experiment["version"]},
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
        else:
            logger.info(
                f'Import experiment {experiment["name"]}.{experiment["version"]}'
            )

        # Get data related to experiment to import.
        algos = src_database.read("algo", {"experiment": experiment["_id"]})
        trials = [
            Trial(**data)
            for data in src_database.read("trials", {"experiment": experiment["_id"]})
        ]

        # Generate new experiment ID
        new_experiment_id = last_experiment_id + 1
        last_experiment_id = new_experiment_id
        experiment["_id"] = new_experiment_id
        # Update algos and trials: set new experiment ID and remove algo/trials database IDs,
        # so that new IDs will be generated at insertion.
        # Trial parents are identified using trial identifier (trial.id)
        # which is not related to trial database ID (trial.id_override).
        # So, we can safely remove trial database ID.
        for algo in algos:
            algo["experiment"] = new_experiment_id
            del algo["_id"]
        for trial in trials:
            trial.experiment = new_experiment_id
            trial.id_override = None

        # Write data
        data_to_add.setdefault(COL_EXPERIMENTS, []).append(experiment)
        data_to_add.setdefault(COL_ALGOS, []).extend(algos)
        data_to_add.setdefault(COL_TRIALS, []).extend(
            trial.to_dict() for trial in trials
        )
    _describe_import_progress(
        STEP_CHECK_SRC_EXPERIMENTS,
        len(experiments),
        len(experiments),
        progress_callback,
    )

    return queries_to_delete, data_to_add


def _execute_import(
    dst_db, queries_to_delete: dict, data_to_add: dict, progress_callback=None
):
    """Execute import
    :param dst_db: destination database where to apply changes
    :param queries_to_delete: dictionary mapping a collection name to a list of queries to use
        to find and delete data
    :param data_to_add: dictionary mapping a collection name to a list of data to add
    """

    total_queries = sum(len(queries) for queries in queries_to_delete.values())
    i_query = 0
    for collection_name, queries in queries_to_delete.items():
        logger.info(
            f"Deleting from {len(queries)} queries into collection {collection_name}"
        )
        for query in queries:
            _describe_import_progress(
                STEP_DELETE_OLD_DATA, i_query, total_queries, progress_callback
            )
            dst_db.remove(collection_name, query)
            i_query += 1
    _describe_import_progress(
        STEP_DELETE_OLD_DATA, total_queries, total_queries, progress_callback
    )
    for i, (collection_name, data) in enumerate(data_to_add.items()):
        _describe_import_progress(
            STEP_INSERT_NEW_DATA, i, len(data_to_add), progress_callback
        )
        logger.info(f"Writing {len(data)} data into collection {collection_name}")
        dst_db.write(collection_name, data)
    _describe_import_progress(
        STEP_INSERT_NEW_DATA, len(data_to_add), len(data_to_add), progress_callback
    )
