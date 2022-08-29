import contextlib
import datetime
import logging
import pickle

import sqlalchemy
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    delete,
    select,
    update,
)
from sqlalchemy.orm import Session, declarative_base

import orion.core
from orion.core.worker.trial import validate_status
from orion.storage.base import (
    BaseStorageProtocol,
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

    uid         = Column(Integer, primary_key=True)
    name        = Column(String(30))
    token       = Column(String(30))
    created_at  = Column(DateTime)
    last_seen   = Column(DateTime)


class Experiment(Base):
    """Defines the Experiment table"""
    __tablename__ = "experiments"

    uid         = Column(Integer, primary_key=True)
    name        = Column(String(30))
    config      = Column(JSON)
    version     = Column(Integer)
    owner_id    = Column(Integer, ForeignKey("user.uid"), nullable=False)
    datetime    = Column(DateTime)


class Trial:
    """Defines the Trial table"""
    __tablename__ = "trial"

    uid             = Column(Integer, primary_key=True)
    experiment_id   = Column(Integer, ForeignKey("experiment.uid"), nullable=False)
    owner_id        = Column(Integer, ForeignKey("user.uid"), nullable=False)
    status          = Column(String(30))
    results         = Column(JSON)
    start_time      = Column(DateTime)
    end_time        = Column(DateTime)
    heartbeat       = Column(DateTime)


class Algo:
    """Defines the Algo table"""
    __tablename__ = "algo"

    uid             = Column(Integer, primary_key=True)
    experiment_id   = Column(Integer, ForeignKey("experiment.uid"), nullable=False)
    owner_id        = Column(Integer, ForeignKey("user.uid"), nullable=False)
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

    def __init__(self, uri):
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

        # engine_from_config
        self.engine = sqlalchemy.create_engine("", echo=True, future=True)

        # Create the schema
        Base.metadata.create_all(self.engine)

        with Session(self.engine) as session:
            stmt = select(User).where(User.token == self.token)
            self.user = session.scalars(stmt).one()

    # Experiment Operations
    # =====================

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""

        with Session(self.engine) as session:
            experiment = Experiment(
                name=config["name"], config=config, onwer_id=self.user.uid, version=0
            )
            session.add(experiment)
            session.commit()

    def delete_experiment(self, experiment, uid):
        uid = get_uid(experiment, uid)

        with Session(self.engine) as session:
            stmt = delete(Experiment).where(Experiment.uid == uid)
            session.execute(stmt)
            session.commit()

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        uid = get_uid(experiment, uid)

        query = True
        if uid is not None:
            query = Experiment.uid == uid

        query = query and self._to_query(Experiment, where)

        with Session(self.engine) as session:
            stmt = select(Experiment).where(query)
            experiment = session.scalars(stmt).one()
            experiment.config = kwargs
            session.commit()

    def fetch_experiments(self, query, selection=None):
        query = self._to_query(query)

        with Session(self.engine) as session:
            stmt = select(Experiment).where(query)
            experiments = session.scalars(stmt).all()

        if selection is not None:
            assert False, "Not Implemented"

        return experiments

    # Benchmarks
    # ==========

    # Trials
    # ======
    def fetch_trials(self, experiment=None, uid=None, where=None):
        uid = get_uid(experiment, uid)

        query = True
        if uid is not None:
            query = Trial.experiment_id == uid

        query = query and self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = select(Trial).where(query)
            return session.scalars(stmt).all()

    def register_trial(self, trial):
        config = trial.to_dict()

        with Session(self.engine) as session:
            stmt = select(Trial).where(Trial.uid == trial._id)
            trial = session.scalars(stmt).one()
            self._set_from_dict(trial, config)
            session.commit()

    def delete_trials(self, experiment=None, uid=None, where=None):
        uid = get_uid(experiment, uid)

        query = True
        if uid is not None:
            query = Trial.experiment_id == uid

        query = query and self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = delete(Trial).where(query)
            session.execute(stmt)
            session.commit()

    def retrieve_result(self, trial, **kwargs):
        return trial

    def get_trial(self, trial=None, uid=None, experiment_uid=None):
        trial_uid, experiment_uid = get_trial_uid_and_exp(trial, uid, experiment_uid)

        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.experiment_id == experiment_uid and Trial.uid == trial_uid
            )
            return session.scalars(stmt).one()

    def update_trials(self, experiment=None, uid=None, where=None, **kwargs):
        uid = get_uid(experiment, uid)
        query = Experiment.uid == uid and self._to_query(Trial, where)

        with Session(self.engine) as session:
            stmt = select(Trial).where(query)
            trials = session.scalars(stmt).all()
            for trial in trials:
                self._set_from_dict(trial, kwargs)
            session.commit()

        return trial

    def update_trial(
        self, trial=None, uid=None, experiment_uid=None, where=None, **kwargs
    ):

        trial_uid, experiment_uid = get_trial_uid_and_exp(trial, uid, experiment_uid)
        query = (
            Trial.uid == trial_uid
            and Trial.experiment_id == experiment_uid
            and self._to_query(where)
        )

        with Session(self.engine) as session:
            stmt = select(Trial).where(query)
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
                Trial.experiment_id == experiment._id
                and Trial.status == "reserved"
                and Trial.heartbeat < threshold
            )
            return session.scalars(stmt).all()

    def push_trial_results(self, trial):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.experiment_id == trial.experiment
                and Trial.uid == trial.id
                and Trial.status == "reserved"
            )
            trial = session.scalars(stmt).one()
            self._set_from_dict(trial, trial.to_dict())
            session.commit()

        return trial

    def set_trial_status(self, trial, status, heartbeat=None, was=None):
        validate_status(status)
        validate_status(was)

        query = Trial.uid == trial.id  # and Trial.experiment_id == trial.experiment
        if was:
            query = query and Trial.status == was

        values = dict(status=status, experiment=trial.experiment)
        if heartbeat:
            values["heartbeat"] = heartbeat

        with Session(self.engine) as session:
            update(Trial).where(query).values(**values)
            session.commit()

    def fetch_pending_trials(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status.in_("interrupted", "new", "suspended")
                and Trial.experiment_id == experiment._id
            )
            return session.scalars(stmt).all()

    def _reserve_trial_postgre(self, experiment):
        now = datetime.datetime.utcnow()

        with Session(self.engine) as session:
            # In PostgrerSQL we can do single query
            stmt = (
                update(Trial)
                .where(
                    True
                    and Trial.status.in_("interrupted", "new", "suspended")
                    and Trial.experiment_id == experiment._id
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
                Trial.status.in_("interrupted", "new", "suspended")
                and Trial.experiment_id == experiment._id
            )
            trial = session.scalars(stmt).one()

            # Update the trial iff the status has not been changed yet
            stmt = (
                update(Trial)
                .where(
                    True
                    and Trial.status == trial.status
                    and Trial.experiment_id == experiment._id
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
            return session.scalars(stmt).all()

    def fetch_noncompleted_trials(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status != "completed" and Trial.experiment_id == experiment._id
            )
            return session.scalars(stmt).all()

    def count_completed(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status == "completed" and Trial.experiment_id == experiment._id
            )
            return session.query(stmt).count()

    def count_broken_trials(self, experiment):
        with Session(self.engine) as session:
            stmt = select(Trial).where(
                Trial.status == "broken" and Trial.experiment_id == experiment._id
            )
            return session.query(stmt).count()

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""

        with Session(self.engine) as session:
            update(Trial).where(
                Trial.uid == trial.id, Trial.status == "reserved"
            ).values(heartbeat=datetime.datetime.utcnow())
            session.commit()

    # Algorithm
    # =========
    def initialize_algorithm_lock(self, experiment_id, algorithm_config):
        with Session(self.engine) as session:
            algo = Algo(
                experiment_id=experiment_id,
                onwer_id=self.user.uid,
                configuration=algorithm_config,
                locked=0,
                heartbeat=datetime.datetime.utcnow(),
            )
            session.add(algo)
            session.commit()

    def release_algorithm_lock(self, experiment=None, uid=None, new_state=None):
        uid = get_uid(experiment, uid)

        values = dict(
            locked=0,
            heartbeat=datetime.datetime.utcnow(),
        )
        if new_state is not None:
            values["state"] = pickle.dumps(new_state)

        with Session(self.engine) as session:
            update(Algo).where(Algo.experiment_id == uid and Algo.locked == 1).values(
                **values
            )

    def get_algorithm_lock_info(self, experiment=None, uid=None):
        """See :func:`orion.storage.base.BaseStorageProtocol.get_algorithm_lock_info`"""
        uid = get_uid(experiment, uid)

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
        self, experiment=None, uid=None, timeout=60, retry_interval=1
    ):
        uid = get_uid(experiment, uid)

        with Session(self.engine) as session:
            now = datetime.datetime.utcnow()

            stmt = (
                update(Algo)
                .where(Algo.experiment_id == uid, Algo.locked == 0)
                .values(locked=1, heartbeat=now)
            )

            session.execute(stmt)
            session.commit()

            algo = select(Algo).where(Algo.experiment_id == uid, Algo.locked == 1)

        if algo is None or algo.heartbead != now:
            return

        algo_state = LockedAlgorithmState(
            state=pickle.loads(algo.state) if algo.state is not None else None,
            configuration=algo.configuration,
            locked=True,
        )

        yield algo_state

        self.release_algorithm_lock(uid, new_state=algo_state.state)

    # Utilities
    # =========
    def _set_from_dict(self, obj, data, rest=None):
        meta = dict()
        while data:
            k, v = data.popitem()

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
        query = True
        for k, v in where.items():

            if hash(table, k):
                query = query and getattr(k) == v
            else:
                log.warning("constrained ignored %s = %s", k, v)

        return query
