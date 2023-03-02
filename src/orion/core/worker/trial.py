# pylint: skip-file
"""
Container class for `Trial` entity
==================================

Describe a particular training run, parameters and results.

"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import warnings
from datetime import timedelta

from orion.core.utils.exceptions import InvalidResult
from orion.core.utils.flatten import unflatten

log = logging.getLogger(__name__)


class AlreadyReleased(Exception):
    """Raised when a trial gets released twice"""


def validate_status(status):
    """
    Verify if given status is valid. Can be one of ``new``, ``reserved``, ``suspended``,
    ``completed``, ``interrupted``, or ``broken``.
    """
    if status is not None and status not in Trial.allowed_stati:
        raise ValueError(f"Given status `{status}` not one of: {Trial.allowed_stati}")


class Trial:
    """Represents an entry in database/trials collection.

    Attributes
    ----------
    experiment: str
       Unique identifier for the experiment that produced this trial.
       Same as an `Experiment._id`.
    id_override: str
        Trial id returned by the database. It should be unique for a given
        set of parameters
    heartbeat: datetime.datetime
        Last time trial was identified as being alive.
    status: str
       Indicates how this trial is currently being used. Can take the following
       values:

       * 'new' : Denotes a fresh set of parameters suggested by an algorithm,
          not yet tried out.
       * 'reserved' : Indicates that this trial is currently being evaluated by
          a worker process, it was a 'new' trial that got selected.
       * 'suspended' : Means that an algorithm decided to stop the evaluation of
          a 'reserved' trial prematurely.
       * 'completed' : is the status of a previously 'reserved' trial that
          successfully got evaluated. `Trial.results` must contain the evaluation.
       * 'interrupted' : Indicates trials that are stopped from being evaluated
          by external *actors* (e.g. cluster timeout, KeyboardInterrupt, killing
          of the worker process).
       * 'broken' : Indicates a trial that was not successfully evaluated for not
          expected reason.
    worker: str
       Corresponds to worker's unique id that handled this trial.
    submit_time: `datetime.datetime`
       When was this trial suggested?
    start_time: `datetime.datetime`
       When was this trial first reserved?
    end_time: `datetime.datetime`
       When was this trial evaluated successfully?
    results: list of `Trial.Result`
       List of evaluated metrics for this particular set of params. One and only
       one of them is necessarily an *objective* function value. The other are
       *constraints*, the value of an expression desired to be larger/equal to 0.
    params: dict of params
       Dict of suggested values for the `Experiment` parameter space.
       Consists a sample to be evaluated.

    """

    @classmethod
    def build(cls, trial_entries):
        """Builder method for a list of trials.

        :param trial_entries: List of trial representation in dictionary form,
           as expected to be saved in a database.

        :returns: a list of corresponding `Trial` objects.
        """
        trials = []
        for entry in trial_entries:
            trials.append(cls(**entry))
        return trials

    class Value:
        """Container for a value object.

        Attributes
        ----------
        name: str
           A possible named for the quality that this is quantifying.
        type: str
           An identifier with semantic importance for **Oríon**. See
           `Param.type` and `Result.type`.
        value: str or numerical
           value suggested for this dimension of the parameter space.

        """

        __slots__ = ("name", "_type", "value")
        allowed_types = ()

        def __init__(self, **kwargs):
            """See attributes of `Value` for possible argument for `kwargs`."""
            for attrname in self.__slots__:
                setattr(self, attrname, None)
            for attrname, value in kwargs.items():
                setattr(self, attrname, value)

            self._ensure_no_ndarray()

        def _ensure_no_ndarray(self):
            """Make sure the current value is not a `numpy.ndarray`."""
            if hasattr(self, "value") and hasattr(self.value, "tolist"):
                self.value = self.value.tolist()

        def to_dict(self):
            """Needed to be able to convert `Value` to `dict` form."""
            ret = dict(name=self.name, type=self.type, value=self.value)
            return ret

        def __eq__(self, other):
            """Test equality based on self.to_dict()"""
            return (
                self.name == other.name
                and self.type == other.type
                and self.value == other.value
            )

        def __str__(self):
            """Represent partially with a string."""
            ret = "{}(name={}, type={}, value={})".format(
                type(self).__name__, repr(self.name), repr(self.type), repr(self.value)
            )
            return ret

        __repr__ = __str__

        @property
        def type(self):
            """For meaning of property type, see `Value.type`."""
            return self._type

        @type.setter
        def type(self, type_):
            if type_ is not None and type_ not in self.allowed_types:
                raise ValueError(
                    f"Given type, {type_}, not one of: {self.allowed_types}"
                )
            self._type = type_

    class Result(Value):
        """Types for a `Result` can be either an evaluation of an 'objective'
        function or of an 'constraint' expression.
        """

        __slots__ = ()
        allowed_types = ("objective", "constraint", "gradient", "statistic", "lie")

    class Param(Value):
        """Types for a `Param` can be either an integer (discrete value),
        floating precision numerical or a categorical expression (e.g. a string).
        """

        __slots__ = ()
        allowed_types = ("integer", "real", "categorical", "fidelity")

    __slots__ = (
        "experiment",
        "_id",
        "_status",
        "worker",
        "_exp_working_dir",
        "heartbeat",
        "submit_time",
        "start_time",
        "end_time",
        "_results",
        "_params",
        "parent",
        "id_override",
    )
    allowed_stati = (
        "new",
        "reserved",
        "suspended",
        "completed",
        "interrupted",
        "broken",
    )

    def __init__(self, **kwargs):
        """See attributes of `Trial` for meaning and possible arguments for `kwargs`."""
        self._params: list[Trial.Param] = []
        self._results: list[Trial.Result] = []

        for attrname in self.__slots__:
            if attrname not in ("_results", "_params"):
                setattr(self, attrname, None)

        self.status = "new"

        # Store the id as an override to support different backends
        self.id_override = kwargs.pop("_id", None)
        kwargs.pop("id", None)

        # NOTE: For backward compatibility with <v0.2.5
        kwargs.pop("id_override", None)

        for attrname, value in kwargs.items():
            if attrname == "parents":
                log.info("Trial.parents attribute is deprecated. Value is ignored.")
            elif attrname == "results":
                attr = getattr(self, attrname)
                for item in value:
                    attr.append(self.Result(**item))
            elif attrname == "params":
                for item in value:
                    self._params.append(self.Param(**item))
            else:
                setattr(self, attrname, value)

    def branch(self, status="new", params=None):
        """Copy the trial and modify given attributes

        The status attributes will be reset as if trial was new.

        Parameters
        ----------
        status: str, optional
            The status of the new trial. Defaults to 'new'.
        params: dict, optional
            Some parameters to update. A subset of params may be passed. Passing
            non-existing params in current trial will lead to a ValueError.
            Defaults to `None`.

        Raises
        ------
        ValueError
            If some parameters are not present in current trial.
        AttributeError
            If some attribute does not exist in Trial objects.
        """
        if params is None:
            params = {}

        params = copy.deepcopy(params)

        config_params = []
        for param in self._params:
            config_param = param.to_dict()
            if param.name in params:
                config_param["value"] = params.pop(param.name)
            config_params.append(config_param)

        if params:
            raise ValueError(f"Some parameters are not part of base trial: {params}")

        return Trial(
            experiment=self.experiment,
            status=status,
            params=config_params,
            parent=self.id,
            exp_working_dir=self.exp_working_dir,
        )

    def to_dict(self):
        """Needed to be able to convert `Trial` to `dict` form."""
        trial_dictionary = dict()

        for attrname in self.__slots__:
            attrname = attrname.lstrip("_")
            trial_dictionary[attrname] = getattr(self, attrname)

        # Overwrite "results" and "params" with list of dictionaries rather
        # than list of Value objects
        trial_dictionary["results"] = list(map(lambda x: x.to_dict(), self.results))
        trial_dictionary["params"] = list(map(lambda x: x.to_dict(), self._params))

        trial_dictionary["id"] = self.id
        id_override = trial_dictionary.pop("id_override", None)
        if id_override:
            trial_dictionary["_id"] = id_override

        return trial_dictionary

    def __str__(self):
        """Represent partially with a string."""
        return "Trial(experiment={}, status={}, params={})".format(
            repr(self.experiment), repr(self._status), self.format_params(self._params)
        )

    __repr__ = __str__

    @property
    def params(self):
        """Parameters of the trial"""
        return unflatten({param.name: param.value for param in self._params})

    @property
    def results(self):
        """List of results of the trial"""
        return self._results

    @results.setter
    def results(self, results):
        """Verify results before setting the property"""
        objective = self._fetch_one_result_of_type("objective", results)

        if objective is None:
            raise InvalidResult(f"No objective found in results: {results}")
        if not isinstance(objective.value, (float, int)):
            raise InvalidResult(
                "Results must contain a type `objective` with type float/int: {}".format(
                    objective
                )
            )

        self._results = results

    def get_working_dir(
        self,
        ignore_fidelity=False,
        ignore_experiment=None,
        ignore_lie=False,
        ignore_parent=False,
    ):
        if self.exp_working_dir is None:
            raise RuntimeError(
                "Cannot infer trial's working_dir because trial.exp_working_dir is not set."
            )
        trial_hash = self.compute_trial_hash(
            self,
            ignore_fidelity=ignore_fidelity,
            ignore_experiment=ignore_experiment,
            ignore_lie=ignore_lie,
            ignore_parent=ignore_parent,
        )
        return os.path.join(self.exp_working_dir, trial_hash)

    @property
    def working_dir(self):
        """Return the current working directory of the trial."""
        return self.get_working_dir()

    @property
    def exp_working_dir(self):
        """Return the current working directory of the experiment."""
        return self._exp_working_dir

    @exp_working_dir.setter
    def exp_working_dir(self, value):
        """Change the current base working directory of the trial."""
        self._exp_working_dir = value

    @property
    def status(self):
        """For meaning of property type, see `Trial.status`."""
        return self._status

    @status.setter
    def status(self, status):
        validate_status(status)
        self._status = status

    @property
    def id(self):
        """Return hash_name which is also the database key ``id``."""
        return self.__hash__()

    @property
    def legacy_id(self):
        """Backward compatible id

        Deprecated and will be removed in v0.4.0.

        This is equivalent to `Trial.id` prior to v0.2.5.
        """
        return self.compute_trial_hash(self, ignore_experiment=False)

    @property
    def objective(self):
        """Return this trial's objective value if it is evaluated, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type("objective")

    @property
    def lie(self):
        """Return this trial's fake objective value if it was set, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type("lie")

    @property
    def gradient(self):
        """Return this trial's gradient value if it is evaluated, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type("gradient")

    @property
    def constraints(self):
        """
        Return this trial's constraints

        Returns
        -------
        A list of ``Trial.Result`` of type 'constraint'
        """
        return self._fetch_results("constraint", self.results)

    @property
    def statistics(self):
        """
        Return this trial's statistics

        Returns
        -------
        A list of ``Trial.Result`` de type 'statistic'
        """
        return self._fetch_results("statistic", self.results)

    @property
    def hash_name(self):
        """Generate a unique name with an md5sum hash for this `Trial`.

        .. note:: Two trials that have the same `params` must have the same `hash_name`.
        """
        return self.compute_trial_hash(self, ignore_fidelity=False)

    @property
    def hash_params(self):
        """Generate a unique param md5sum hash for this `Trial`.

        .. note:: The params contributing to the hash do not include the fidelity.
        """
        return self.compute_trial_hash(
            self, ignore_fidelity=True, ignore_lie=True, ignore_parent=True
        )

    def __eq__(self, other):
        """Whether two trials are equal is based on id alone.

        This includes params, experiment, parent and lie. All other attributes of the
        trials are ignored when comparing them.
        """
        return self.id == other.id

    def __hash__(self):
        """Return the hashname for this trial"""
        return self.hash_name

    @property
    def full_name(self):
        """Generate a unique name using the full definition of parameters."""
        if not self._params or not self.experiment:
            raise ValueError(
                "Cannot distinguish this trial, as 'params' or 'experiment' "
                "have not been set."
            )
        return self.format_values(self._params, sep="-").replace("/", ".")

    @property
    def duration(self):
        """Return trial duration as a timedelta() object"""
        execution_interval = self.execution_interval
        if execution_interval:
            from_time, to_time = execution_interval
            return to_time - from_time
        else:
            return timedelta()

    @property
    def execution_interval(self):
        """Return execution interval, or None if unavailable"""
        if self.start_time:
            if self.end_time:
                return self.start_time, self.end_time
            elif self.heartbeat:
                return self.start_time, self.heartbeat
        return None

    def _repr_values(self, values, sep=","):
        """Represent with a string the given values."""
        return Trial.format_values(values, sep)

    def params_repr(self, sep=",", ignore_fidelity=False):
        """Represent with a string the parameters contained in this `Trial` object."""
        return Trial.format_params(self._params, sep)

    @staticmethod
    def format_values(values, sep=","):
        """Represent with a string the given values."""
        return sep.join(map(lambda value: "{0.name}:{0.value}".format(value), values))

    @staticmethod
    def format_params(params, sep=",", ignore_fidelity=False):
        """Represent with a string the parameters contained in this `Trial` object."""
        if ignore_fidelity:
            params = [x for x in params if x.type != "fidelity"]
        else:
            params = params
        return Trial.format_values(params, sep)

    @staticmethod
    def compute_trial_hash(
        trial: Trial,
        ignore_fidelity=False,
        ignore_experiment=None,
        ignore_lie=False,
        ignore_parent=False,
    ):
        """Generate a unique param md5sum hash for a given `Trial`"""
        if not trial._params and not trial.experiment:
            raise ValueError(
                "Cannot distinguish this trial, as 'params' or 'experiment' "
                "have not been set."
            )

        params = Trial.format_params(trial._params, ignore_fidelity=ignore_fidelity)

        if ignore_experiment is not None:
            warnings.warn(
                "Argument ignore_experiment is deprecated and will be removed in v0.3.0. "
                "Trial.id does not include experiment id since release v0.2.5.",
                DeprecationWarning,
            )
        else:
            ignore_experiment = True

        experiment_repr = ""
        if not ignore_experiment and trial.experiment is not None:
            experiment_repr = str(trial.experiment)

        lie_repr = ""
        if not ignore_lie and trial.lie:
            lie_repr = Trial.format_values([trial.lie])

        # TODO: When implementing TrialClient, we should compute the hash of the parent
        #       based on the same ignore_ attributes. For now we use the full id of the parent.
        parent_repr = ""
        if not ignore_parent and trial.parent is not None:
            parent_repr = str(trial.parent)

        return hashlib.md5(
            (params + experiment_repr + lie_repr + parent_repr).encode("utf-8")
        ).hexdigest()

    def _fetch_results(self, type, results):
        """Fetch results for the given type"""
        return [result for result in results if result.type == type]

    def _fetch_one_result_of_type(self, result_type, results=None):
        if results is None:
            results = self.results

        value = self._fetch_results(result_type, results)

        if not value:
            return None

        if len(value) > 1:
            log.warning("Found multiple results of '%s' type:\n%s", result_type, value)
            log.warning(
                "Multi-objective optimization is not currently supported.\n"
                "Optimizing according to the first one only: %s",
                value[0],
            )

        return value[0]


class TrialCM:
    __slots__ = ("_cm_experiment", "_cm_trial")

    def __init__(self, experiment, trial):
        self._cm_experiment = experiment
        self._cm_trial = trial

    def __getattribute__(self, name):
        if name in {"_cm_experiment", "_cm_trial", "__enter__", "__exit__"}:
            return object.__getattribute__(self, name)
        return getattr(self._cm_trial, name)

    def __setattr__(self, name, value):
        if name not in {"_cm_experiment", "_cm_trial"}:
            setattr(self._cm_trial, name, value)
        else:
            object.__setattr__(self, name, value)

    def __enter__(self):
        return self._cm_trial

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is KeyboardInterrupt:
                self._cm_experiment.release(self._cm_trial, "interrupted")
            elif exc_type is not None:
                self._cm_experiment.release(self._cm_trial, "broken")
            elif self._cm_trial.status == "reserved":
                self._cm_experiment.release(self._cm_trial)
        except AlreadyReleased as e:
            log.warning(e)
