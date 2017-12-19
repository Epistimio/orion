# -*- coding: utf-8 -*-
# pylint: skip-file
"""
:mod:`metaopt.worker.trial` -- Container class for `Trial` entity
=================================================================

.. module:: trial
   :platform: Unix
   :synopsis: Describe a particular training run, parameters and results

"""

import logging

import six

log = logging.getLogger(__name__)


class Trial(object):
    """Represents an entry in database/trials collection.

    Attributes
    ----------
    experiment : str
       Unique identifier for the experiment taht produced this trial.
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
           An identifier with semantic importance for **MetaOpt**. See
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
            for attrname, value in six.iteritems(kwargs):
                setattr(self, attrname, value)

        def to_dict(self):
            """Needed to be able to convert `Value` to `dict` form."""
            ret = dict(
                name=self.name,
                type=self._type,
                value=self.value
                )
            return ret

        def __str__(self):
            """Represent partially with a string."""
            ret = "{0}(name={1}, type={2}, value={3})".format(
                type(self).__name__, repr(self.name),
                repr(self._type), repr(self.value))
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

        allowed_types = ('objective', 'constraint')

    class Param(Value):
        """Types for a `Param` can be either an integer (discrete value),
        floating precision numerical or a categorical expression (e.g. a string).
        """

        allowed_types = ('int', 'float', 'enum')

    __slots__ = ('experiment', '_id', '_status', 'worker',
                 'submit_time', 'start_time', 'end_time', 'results', 'params')
    allowed_stati = ('new', 'reserved', 'suspended', 'completed', 'interrupted', 'broken')
    NoID = None

    def __init__(self, **kwargs):
        """See attributes of `Trial` for meaning and possible arguments for `kwargs`."""
        for attrname in self.__slots__:
            if attrname in ('results', 'params'):
                setattr(self, attrname, list())
            else:
                setattr(self, attrname, None)

        self.status = 'new'

        for attrname, value in six.iteritems(kwargs):
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

        # Trial object should not be able to set a value
        # (whatever value) to its `_id` on its own, but is should just
        # expect that it could have one
        trial_dictionary.pop('id')  # Pop invalid name for database id key
        if self.is_registered:
            trial_dictionary['_id'] = self.id

        return trial_dictionary

    def __str__(self):
        """Represent partially with a string."""
        ret = "Trial(experiment={0}, status={1}, params.value={2})".format(
            repr(self.experiment), repr(self._status), [p.value for p in self.params])
        return ret

    __repr__ = __str__

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
        """Return database key `_id`."""
        return self._id

    @property
    def objective(self):
        """Return this trial's objective value if it is evaluated, else None.

        :rtype: `Trial.Result`
        """
        value = [result for result in self.results
                 if result.type == 'objective']

        if not value:
            return None

        if len(value) > 1:
            log.warning("Found multiple results of objective function type:\n%s",
                        value)
            log.warning("Multi-objective optimization is not currently supported.\n"
                        "Optimizing according to the first one only: %s", value[0])

        return value[0]

    @property
    def is_registered(self):
        """Check whether `Trial` is registered in database based on `_id` value."""
        return self._id is not self.NoID
