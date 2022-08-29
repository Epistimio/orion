import contextlib
import datetime
import getpass
import logging
import pickle
import uuid
from copy import deepcopy

import sqlalchemy
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    delete,
    select,
    update,
)
from sqlalchemy.exc import DBAPIError, NoResultFound
from sqlalchemy.orm import Session, declarative_base

import orion.core
from orion.core.io.database import DuplicateKeyError
from orion.core.worker.trial import Trial as OrionTrial
from orion.core.worker.trial import validate_status
from orion.storage.base import (
    BaseStorageProtocol,
    FailedUpdate,
    LockedAlgorithmState,
    get_trial_uid_and_exp,
    get_uid,
)

log = logging.getLogger(__name__)

Base = declarative_base()

# fmt: off
class User(Base):
    """Defines the User table"""
    __tablename__ = "users"

    _id         = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(30))
    token       = Column(String(32))
    created_at  = Column(DateTime)
    last_seen   = Column(DateTime)


class Experiment(Base):
    """Defines the Experiment table"""
    __tablename__ = "experiments"

    _id         = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(30))
    meta        = Column(JSON)          # metadata field is reserved
    version     = Column(Integer)
    owner_id    = Column(Integer, ForeignKey("users._id"), nullable=False)
    datetime    = Column(DateTime)
    algorithms  = Column(JSON)
    remaining   = Column(JSON)
    space       = Column(JSON)

    __table_args__ = (
        UniqueConstraint('name', 'owner_id', name='_one_name_per_owner'),
    )


class Trial(Base):
    """Defines the Trial table"""
    __tablename__ = "trials"

    _id             = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id   = Column(Integer, ForeignKey("experiments._id"), nullable=False)
    owner_id        = Column(Integer, ForeignKey("users._id"), nullable=False)
    status          = Column(String(30))
    results         = Column(JSON)
    start_time      = Column(DateTime)
    end_time        = Column(DateTime)
    heartbeat       = Column(DateTime)
    parent          = Column(Integer, ForeignKey("experiments._id"))
    params          = Column(JSON)
    worker          = Column(JSON)
    submit_time     = Column(String(30))
    exp_working_dir = Column(String(30))
    id              = Column(String(30))

    __table_args__ = (
        UniqueConstraint('experiment_id', 'id', name='_one_trial_hash_per_experiment'),
    )


class Algo(Base):
    """Defines the Algo table"""
    __tablename__ = "algo"

    _id             = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id   = Column(Integer, ForeignKey("experiments._id"), nullable=False)
    owner_id        = Column(Integer, ForeignKey("users._id"), nullable=False)
    configuration   = Column(JSON)
    locked          = Column(Integer)
    state           = Column(JSON)
    heartbeat       = Column(DateTime)
# fmt: on


class SQLAlchemy(BaseStorageProtocol):  # noqa: F811
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend

    Parameters
    ----------
    uri: str
        PostgreSQL backend to use for storage; the format is as follow
         `protocol://[username:password@]host1[:port1][,...hostN[:portN]]][/[database][?options]]`

    """

    def __init__(self, uri, token=None, **kwargs):
        # dialect+driver://username:password@host:port/database
        #
        # postgresql://scott:tiger@localhost/mydatabase
        # postgresql+psycopg2://scott:tiger@localhost/mydatabase
        # postgresql+pg8000://scott:tiger@localhost/mydatabase
        #
        # mysql://scott:tiger@localhost/foo
        # mysql+mysqldb://scott:tiger@localhost/foo
        # mysql+pymysql://scott:tiger@localhost/foo
        #
        # sqlite:///foo.db
        # sqlite://             # in memory

        self.uri = uri
        if uri == "":
            uri = "sqlite://"

        # engine_from_config
        self.engine = sqlalchemy.create_engine(uri, echo=True, future=True)

        # Create the schema
        Base.metadata.create_all(self.engine)

        self.token = token
        self.user_id = None
        self.user = None
        self._connect(token)

    def _connect(self, token):
        if token is not None and token != "":
            with Session(self.engine) as session:
                stmt = select(User).where(User.token == self.token)
                self.user = session.scalars(stmt).one()

                self.user_id = self.user._id
        else:
            # Local database, create a default user
            user = getpass.getuser()
            now = datetime.datetime.utcnow()

            with Session(self.engine) as session:
                self.user = User(
                    name=user,
                    token=uuid.uuid5(uuid.NAMESPACE_OID, user).hex,
                    created_at=now,
                    last_seen=now,
                )
                session.add(self.user)
                session.commit()

                assert self.user._id > 0
                self.user_id = self.user._id

    def __getstate__(self):
        return dict(
            uri=self.uri,
            token=self.token,
        )

    def __setstate__(self, state):
        self.uri = state["uri"]
        self.token = state["token"]
        self.engine = sqlalchemy.create_engine(self.uri, echo=True, future=True)
        self._connect(self.token)

    # Experiment Operations
    # =====================

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        config = deepcopy(config)

        try:
            with Session(self.engine) as session:
                experiment = Experiment(
                    owner_id=self.user_id,
                    version=0,
                )

                config["meta"] = config.pop("metadata")

                # old way
                # if 'space' not in config:
                #    config['space'] = config['meta'].pop('priors', dict())

                self._set_from_dict(experiment, config, "remaining")

                session.add(experiment)
                session.commit()

        except DBAPIError:
            raise DuplicateKeyError()

    def delete_experiment(self, experiment=None, uid=None):
        uid = get_uid(experiment, uid)

        with Session(self.engine) as session:
            stmt = delete(Experiment).where(Experiment._id == uid)
            session.execute(stmt)
            session.commit()

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        uid = get_uid(experiment, uid)

        query = True
        if where is None:
            where = dict()

        if uid is not None:
            where["_id"] = uid

        query = self._to_query(Experiment, where)

        with Session(self.engine) as session:
            stmt = select(Experiment).where(*query)
            experiment = session.scalars(stmt).one()

            metadata = kwargs.pop("metadata", dict())
            self._set_from_dict(experiment, kwargs, "remaining")
            experiment.meta.update(metadata)

            session.commit()

    def fetch_experiments(self, query, selection=None):
        query = self._to_query(Experiment, query)

        with Session(self.engine) as session:
            stmt = select(Experiment).where(*query)
            experiments = session.scalars(stmt).all()

        if selection is not None:
            assert False, "Not Implemented"

        return [self._to_experiment(exp) for exp in experiments]

    # Benchmarks
    # ==========

    # Trials
    # ======
    def fetch_trials(self, experiment=None, uid=None, where=None):
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        if uid is not None:
            where["experiment_id"] = uid

        query = self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = select(Trial).where(*query)
            return session.scalars(stmt).all()

    def register_trial(self, trial):
        config = trial.to_dict()

        try:
            with Session(self.engine) as session:
                experiment_id = config.pop("experiment", None)

                db_trial = Trial(experiment_id=experiment_id, owner_id=self.user_id)

                self._set_from_dict(db_trial, config)

                session.add(db_trial)
                session.commit()

                session.refresh(db_trial)

                return OrionTrial(**self._to_trial(db_trial))
        except DBAPIError:
            raise DuplicateKeyError()

    def delete_trials(self, experiment=None, uid=None, where=None):
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        if uid is not None:
            where["experiment_id"] = uid

        query = self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = delete(Trial).where(*query)
            count = session.execute(stmt)
            session.commit()

            return count.rowcount

    def retrieve_result(self, trial, **kwargs):
        return trial

    def get_trial(self, trial=None, uid=None, experiment_uid=None):
        trial_uid, experiment_uid = get_trial_uid_and_exp(trial, uid, experiment_uid)

        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.experiment_id == experiment_uid,
                Trial.id == trial_uid,
            )
            trial = session.scalars(stmt).one()

        return OrionTrial(**self._to_trial(trial))

    def update_trials(self, experiment=None, uid=None, where=None, **kwargs):
        uid = get_uid(experiment, uid)

        if where is None:
            where = dict()

        where["experiment_id"] = uid
        query = self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = select(Trial).where(*query)
            trials = session.scalars(stmt).all()

            for trial in trials:
                self._set_from_dict(trial, kwargs)

            session.commit()

            return len(trials)

    def update_trial(
        self, trial=None, uid=None, experiment_uid=None, where=None, **kwargs
    ):

        trial_uid, experiment_uid = get_trial_uid_and_exp(trial, uid, experiment_uid)

        if where is None:
            where = dict()

        # THIS IS NOT THE UNIQUE ID OF THE TRIAL
        where["id"] = trial_uid
        where["experiment_id"] = experiment_uid
        query = self._to_query(Trial, where)

        print()
        print(query)
        print()

        with Session(self.engine) as session:
            stmt = select(Trial).where(*query)
            trial = session.scalars(stmt).one()

            self._set_from_dict(trial, kwargs)
            session.commit()

        return trial

    def fetch_lost_trials(self, experiment):
        heartbeat = orion.core.config.worker.heartbeat
        threshold = datetime.datetime.utcnow() - datetime.timedelta(
            seconds=heartbeat * 5
        )

        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.experiment_id == experiment._id,
                Trial.status == "reserved",
                Trial.heartbeat < threshold,
            )
            return session.scalars(stmt).all()

    def push_trial_results(self, trial):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.experiment_id == trial.experiment,
                Trial._id == trial.id,
                Trial.status == "reserved",
            )
            trial = session.scalars(stmt).one()
            self._set_from_dict(trial, trial.to_dict())
            session.commit()

        return trial

    def set_trial_status(self, trial, status, heartbeat=None, was=None):
        heartbeat = heartbeat or datetime.datetime.utcnow()
        was = was or trial.status

        validate_status(status)
        validate_status(was)

        query = [
            Trial.id == trial.id,
            Trial.experiment_id == trial.experiment,
            Trial.status == was,
        ]

        values = dict(status=status)
        if heartbeat:
            values["heartbeat"] = heartbeat

        with Session(self.engine) as session:
            stmt = update(Trial).where(*query).values(**values)
            result = session.execute(stmt)
            session.commit()

            if result.rowcount == 1:
                trial.status = status
            else:
                raise FailedUpdate()

    def fetch_pending_trials(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status.in_(("interrupted", "new", "suspended")),
                Trial.experiment_id == experiment._id,
            )
            return session.scalars(stmt).all()

    def _reserve_trial_postgre(self, experiment):
        now = datetime.datetime.utcnow()

        with Session(self.engine) as session:
            # In PostgrerSQL we can do single query
            stmt = (
                update(Trial)
                .where(
                    Trial.status.in_(("interrupted", "new", "suspended")),
                    Trial.experiment_id == experiment._id,
                )
                .values(
                    status="reserved",
                    start_time=now,
                    heartbeat=now,
                )
                .limit(1)
                .returning()
            )
            trial = session.scalar(stmt)
            return trial

    def reserve_trial(self, experiment):
        if False:
            return self._reserve_trial_postgre(experiment)

        now = datetime.datetime.utcnow()

        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status.in_(("interrupted", "new", "suspended")),
                Trial.experiment_id == experiment._id,
            )
            try:
                trial = session.scalars(stmt).one()
            except NoResultFound:
                return None

            # Update the trial iff the status has not been changed yet
            stmt = (
                update(Trial)
                .where(
                    Trial.status == trial.status,
                    Trial.experiment_id == experiment._id,
                )
                .values(
                    status="reserved",
                    start_time=now,
                    heartbeat=now,
                )
            )

            session.execute(stmt)

            stmt = select(Trial).where(Trial.experiment_id == experiment._id)
            trial = session.scalars(stmt).one()

            # time needs to match, could have been reserved by another worker
            if trial.status == "reserved" and trial.heartbeat == now:
                return trial

            return None

    def fetch_trials_by_status(self, experiment, status):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status == status and Trial.experiment_id == experiment._id
            )
            return [
                OrionTrial(**self._to_trial(trial))
                for trial in session.scalars(stmt).all()
            ]

    def fetch_noncompleted_trials(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status != "completed",
                Trial.experiment_id == experiment._id,
            )
            return session.scalars(stmt).all()

    def count_completed_trials(self, experiment):
        with Session(self.engine) as session:
            return (
                session.query(Trial)
                .filter(
                    Trial.status == "completed",
                    Trial.experiment_id == experiment._id,
                )
                .count()
            )

    def count_broken_trials(self, experiment):
        with Session(self.engine) as session:
            return (
                session.query(Trial)
                .filter(
                    Trial.status == "broken",
                    Trial.experiment_id == experiment._id,
                )
                .count()
            )

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""

        with Session(self.engine) as session:
            stmt = (
                update(Trial)
                .where(
                    Trial._id == trial.id_override,
                    Trial.status == "reserved",
                )
                .values(heartbeat=datetime.datetime.utcnow())
            )

            cursor = session.execute(stmt)
            session.commit()

            if cursor.rowcount <= 0:
                raise FailedUpdate()

    # Algorithm
    # =========
    def initialize_algorithm_lock(self, experiment_id, algorithm_config):
        with Session(self.engine) as session:
            algo = Algo(
                experiment_id=experiment_id,
                owner_id=self.user._id,
                configuration=algorithm_config,
                locked=0,
                heartbeat=datetime.datetime.utcnow(),
            )
            session.add(algo)
            session.commit()

    def release_algorithm_lock(self, experiment=None, _id=None, new_state=None):
        _id = get_uid(experiment, _id)

        values = dict(
            locked=0,
            heartbeat=datetime.datetime.utcnow(),
        )
        if new_state is not None:
            values["state"] = pickle.dumps(new_state)

        with Session(self.engine) as session:
            update(Algo).where(Algo.experiment_id == _id and Algo.locked == 1).values(
                **values
            )

    def get_algorithm_lock_info(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.get_algorithm_lock_info`"""
        _id = get_uid(experiment, uid)

        with Session(self.engine) as session:
            stmt = select(Algo).where(Algo.experiment_id == uid)
            algo = session.scalar(stmt).one()

        return LockedAlgorithmState(
            state=pickle.loads(algo.state) if algo.state is not None else None,
            configuration=algo.configuration,
            locked=algo.locked,
        )

    def delete_algorithm_lock(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.delete_algorithm_lock`"""
        uid = get_uid(experiment, uid)

        with Session(self.engine) as session:
            stmt = delete(Algo).where(Algo.experiment_id == uid)
            session.execute(stmt)
            session.commit()

    def _acquire_algorithm_lock_postgre(
        self, experiment=None, uid=None, timeout=60, retry_interval=1
    ):
        with Session(self.engine) as session:
            now = datetime.datetime.utcnow()

            stmt = (
                update(Algo)
                .where(Algo.experiment_id == uid, Algo.locked == 0)
                .values(locked=1, heartbeat=now)
                .returning()
            )

            algo = session.scalar(stmt).one()
            session.commit()
            return algo

    @contextlib.contextmanager
    def acquire_algorithm_lock(
        self, experiment=None, _id=None, timeout=60, retry_interval=1
    ):
        _id = get_uid(experiment, _id)

        with Session(self.engine) as session:
            now = datetime.datetime.utcnow()

            stmt = (
                update(Algo)
                .where(Algo.experiment_id == _id, Algo.locked == 0)
                .values(locked=1, heartbeat=now)
            )

            session.execute(stmt)
            session.commit()

            stmt = select(Algo).where(Algo.experiment_id == _id, Algo.locked == 1)
            algo = session.scalar(stmt)

        if algo is None or algo.heartbeat != now:
            yield None
            return

        algo_state = LockedAlgorithmState(
            state=pickle.loads(algo.state) if algo.state is not None else None,
            configuration=algo.configuration,
            locked=True,
        )

        yield algo_state

        self.release_algorithm_lock(_id, new_state=algo_state.state)

    # Utilities
    # =========
    def _set_from_dict(self, obj, data, rest=None):
        data = deepcopy(data)
        meta = dict()
        while data:
            k, v = data.popitem()

            if v is None:
                continue

            if hasattr(obj, k):
                setattr(obj, k, v)
            else:
                meta[k] = v

        if meta and rest:
            setattr(obj, rest, meta)
            return

        if meta:
            log.warning("Data was discarded %s", meta)

    def _to_query(self, table, where):
        query = []

        for k, v in where.items():
            if hasattr(table, k):
                query.append(getattr(table, k) == v)
            else:
                log.warning("constrained ignored %s = %s", k, v)

        return query

    def _to_experiment(self, experiment):
        exp = deepcopy(experiment.__dict__)
        exp["metadata"] = exp.pop("meta", {})
        exp.pop("_sa_instance_state")
        exp.pop("owner_id")
        exp.pop("datetime")

        none_keys = []
        for k, v in exp.items():
            if v is None:
                none_keys.append(k)

        for k in none_keys:
            exp.pop(k)

        rest = exp.pop("remaining", {})
        if rest is None:
            rest = {}

        exp.update(rest)

        return exp

    def _to_trial(self, trial):
        trial = deepcopy(trial.__dict__)
        trial.pop("_sa_instance_state")
        trial["experiment"] = trial.pop("experiment_id")
        trial.pop("owner_id")
        return trial
