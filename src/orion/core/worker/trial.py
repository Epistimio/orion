# -*- coding: utf-8 -*-
# pylint: skip-file
"""
:mod:`orion.core.worker.trial` -- Container class for `Trial` entity
====================================================================

.. module:: trial
   :platform: Unix
   :synopsis: Describe a particular training run, parameters and results

"""
import hashlib
import logging

log = logging.getLogger(__name__)


class Trial(object):
    """Represents an entry in database/trials collection.

    Attributes
    ----------
    experiment : str
       Unique identifier for the experiment that produced this trial.
       Same as an `Experiment._id`.
    status : str
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
    worker : str
       Corresponds to worker's unique id that handled this trial.
    submit_time : `datetime.datetime`
       When was this trial suggested?
    start_time : `datetime.datetime`
       When was this trial first reserved?
    end_time : `datetime.datetime`
       When was this trial evaluated successfully?
    results : list of `Trial.Result`
       List of evaluated metrics for this particular set of params. One and only
       one of them is necessarily an *objective* function value. The other are
       *constraints*, the value of an expression desired to be larger/equal to 0.
    params : list of `Trial.Param`
       List of suggested values for the `Experiment` parameter space.
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

    class Value(object):
        """Container for a value object.

        Attributes
        ----------
        name : str
           A possible named for the quality that this is quantifying.
        type : str
           An identifier with semantic importance for **Oríon**. See
           `Param.type` and `Result.type`.
        value : str or numerical
           value suggested for this dimension of the parameter space.

        """

        __slots__ = ('name', '_type', 'value')
        allowed_types = ()

        def __init__(self, **kwargs):
            """See attributes of `Value` for possible argument for `kwargs`."""
            for attrname in self.__slots__:
                setattr(self, attrname, None)
            for attrname, value in kwargs.items():
                setattr(self, attrname, value)

        def to_dict(self):
            """Needed to be able to convert `Value` to `dict` form."""
            ret = dict(
                name=self.name,
                type=self.type,
                value=self.value
                )
            return ret

        def __eq__(self, other):
            """Test equality based on self.to_dict()"""
            return self.name == other.name and self.type == other.type and self.value == other.value

        def __str__(self):
            """Represent partially with a string."""
            ret = "{0}(name={1}, type={2}, value={3})".format(
                type(self).__name__, repr(self.name),
                repr(self.type), repr(self.value))
            return ret

        __repr__ = __str__

        @property
        def type(self):
            """For meaning of property type, see `Value.type`."""
            return self._type

        @type.setter
        def type(self, type_):
            if type_ is not None and type_ not in self.allowed_types:
                raise ValueError("Given type, {0}, not one of: {1}".format(
                    type_, self.allowed_types))
            self._type = type_

    class Result(Value):
        """Types for a `Result` can be either an evaluation of an 'objective'
        function or of an 'constraint' expression.
        """

        __slots__ = ()
        allowed_types = ('objective', 'constraint', 'gradient', 'statistic', 'lie')

    class Param(Value):
        """Types for a `Param` can be either an integer (discrete value),
        floating precision numerical or a categorical expression (e.g. a string).
        """

        __slots__ = ()
        allowed_types = ('integer', 'real', 'categorical', 'fidelity')

    __slots__ = ('experiment', 'index', '_id', '_status', 'worker',
                 'submit_time', 'start_time', 'end_time', '_results', 'params', 'parents')
    allowed_stati = ('new', 'reserved', 'suspended', 'completed', 'interrupted', 'broken')

    def __init__(self, **kwargs):
        """See attributes of `Trial` for meaning and possible arguments for `kwargs`."""
        for attrname in self.__slots__:
            if attrname in ('_results', 'params', 'parents'):
                setattr(self, attrname, list())
            else:
                setattr(self, attrname, None)

        self.status = 'new'

        # Remove useless item
        kwargs.pop('_id', None)

        for attrname, value in kwargs.items():
            if attrname == 'results':
                attr = getattr(self, attrname)
                for item in value:
                    attr.append(self.Result(**item))
            elif attrname == 'params':
                attr = getattr(self, attrname)
                for item in value:
                    attr.append(self.Param(**item))
            else:
                setattr(self, attrname, value)

    def to_dict(self):
        """Needed to be able to convert `Trial` to `dict` form."""
        trial_dictionary = dict()

        for attrname in self.__slots__:
            attrname = attrname.lstrip("_")
            trial_dictionary[attrname] = getattr(self, attrname)

        # Overwrite "results" and "params" with list of dictionaries rather
        # than list of Value objects
        for attrname in ('results', 'params'):
            trial_dictionary[attrname] = list(map(lambda x: x.to_dict(),
                                                  getattr(self, attrname)))

        trial_dictionary['_id'] = trial_dictionary.pop('id')

        return trial_dictionary

    def __str__(self):
        """Represent partially with a string."""
        ret = "Trial(experiment={0}, status={1},\n      params={2})".format(
            repr(self.experiment), repr(self._status), self.params_repr(sep=('\n' + ' ' * 13)))
        return ret

    __repr__ = __str__

    @property
    def results(self):
        """List of results of the trial"""
        return self._results

    @results.setter
    def results(self, results):
        """Verify results before setting the property"""
        objective = self._fetch_one_result_of_type('objective', results)

        if objective is None:
            raise ValueError('No objective found in results: {}'.format(results))
        if not isinstance(objective.value, (float, int)):
            raise ValueError(
                'Results must contain a type `objective` with type float/int: {}'.format(objective))

        self._results = results

    @property
    def status(self):
        """For meaning of property type, see `Trial.status`."""
        return self._status

    @status.setter
    def status(self, status):
        if status is not None and status not in self.allowed_stati:
            raise ValueError("Given status, {0}, not one of: {1}".format(
                status, self.allowed_stati))
        self._status = status

    @property
    def id(self):
        """Return hash_name which is also the database key `_id`."""
        return self.__hash__()

    @property
    def objective(self):
        """Return this trial's objective value if it is evaluated, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type('objective')

    @property
    def lie(self):
        """Return this trial's fake objective value if it was set, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type('lie')

    @property
    def gradient(self):
        """Return this trial's gradient value if it is evaluated, else None.

        :rtype: `Trial.Result`
        """
        return self._fetch_one_result_of_type('gradient')

    def _repr_values(self, values, sep=','):
        """Represent with a string the given values."""
        return sep.join(map(lambda value: "{0.name}:{0.value}".format(value), values))

    def params_repr(self, sep=','):
        """Represent with a string the parameters contained in this `Trial` object."""
        return self._repr_values(self.params, sep)

    @property
    def hash_name(self):
        """Generate a unique name with an md5sum hash for this `Trial`.

        .. note:: Two trials that have the same `params` must have the same `hash_name`.
        """
        if not self.params and not self.experiment:
            raise ValueError("Cannot distinguish this trial, as 'params' or 'experiment' "
                             "have not been set.")
        params_repr = self.params_repr()
        experiment_repr = str(self.experiment)
        lie_repr = self._repr_values([self.lie]) if self.lie else ""
        return hashlib.md5((params_repr + experiment_repr + lie_repr).encode('utf-8')).hexdigest()

    def __hash__(self):
        """Return the hashname for this trial"""
        return self.hash_name

    @property
    def full_name(self):
        """Generate a unique name using the full definition of parameters."""
        if not self.params or not self.experiment:
            raise ValueError("Cannot distinguish this trial, as 'params' or 'experiment' "
                             "have not been set.")
        return self.params_repr(sep='-').replace('/', '.')

    def _fetch_one_result_of_type(self, result_type, results=None):
        if results is None:
            results = self.results

        value = [result for result in results if result.type == result_type]

        if not value:
            return None

        if len(value) > 1:
            log.warning("Found multiple results of '%s' type:\n%s",
                        result_type, value)
            log.warning("Multi-objective optimization is not currently supported.\n"
                        "Optimizing according to the first one only: %s", value[0])

        return value[0]
