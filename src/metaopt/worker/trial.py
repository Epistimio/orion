# -*- coding: utf-8 -*-
"""
:mod:`metaopt.worker.trial` -- Container class for `Trial` entity
=================================================================

.. module:: trial
   :platform: Unix
   :synopsis: Describe a particular training run, hyperparameters and results

"""

import six


class Trial(object):
    """Represents an entry in database/trials collection."""

    @classmethod
    def build(cls, trial_entries):
        trials = []
        for entry in trial_entries:
            trials.append(cls(**entry))
        return trials

    class Value(object):
        __slots__ = ('name', '_type', 'value')
        allowed_types = ()

        def __init__(self, **kwargs):
            for attrname in self.__slots__:
                setattr(self, attrname, None)
            for attrname, value in six.iteritems(kwargs):
                setattr(self, attrname, value)

        def __iter__(self):
            for attrname in self.__slots__:
                if attrname == '_type':
                    yield ('type', self._type)
                else:
                    yield (attrname, getattr(self, attrname))

        def __str__(self):
            ret = "{0}(name={1}, type={2}, value={3})".format(
                type(self).__name__, repr(self.name),
                repr(self._type), repr(self.value))
            return ret

        @property
        def type(self):
            return self._type

        @type.setter
        def type(self, type_):
            if type_ is not None and type_ not in self.allowed_types:
                raise ValueError("Given type, {0}, not one of: {1}".format(
                    type_, self.allowed_types))
            self._type = type_

    class Result(Value):
        allowed_types = ('objective', 'constraint')

    class Param(Value):
        allowed_types = ('int', 'float', 'enum')

    __slots__ = ('exp_name', 'user', '_status', 'worker',
                 'submit_time', 'start_time', 'end_time', 'results', 'params')
    allowed_stati = ('new', 'reserved', 'suspended', 'completed', 'broken')

    def __init__(self, exp_name, user, **kwargs):
        for attrname in self.__slots__:
            if attrname in ('results', 'params'):
                setattr(self, attrname, list())
            else:
                setattr(self, attrname, None)

        self.exp_name = exp_name
        self.user = user
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

    def __iter__(self):
        for attrname in self.__slots__:
            if attrname in ('results', 'params'):
                yield (attrname, list(map(dict, getattr(self, attrname))))
            elif attrname == '_status':
                yield ('status', self._status)
            else:
                yield (attrname, getattr(self, attrname))

    def __str__(self):
        ret = "Trial(exp_name={0}, status={1}, params.value={2})".format(
            repr(self.exp_name), repr(self._status), [p.value for p in self.params])
        return ret

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status):
        if status is not None and status not in self.allowed_stati:
            raise ValueError("Given status, {0}, not one of: {1}".format(
                status, self.allowed_stati))
        self._status = status
