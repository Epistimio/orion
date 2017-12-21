# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`metaopt.core.io.space_builder` -- Create Space objects from configuration
===============================================================================

.. module:: space_builder
   :platform: Unix
   :synopsis: Functions which build `Dimension` and `Space` objects for
      defining problem's search space.

"""

import numbers
import re

from scipy.stats import distributions as sp_dists
import six

from metaopt.algo.space import (Categorical, Integer, Real)


USERCONFIG_KEYWORD = 'mopt~'
USERARGS_SEP = '~'
userconfig_tmpl = None
userargs_tmpl = None
config_method = None


def _check_expr_to_eval(expr):
    if '__' in expr or ';' in expr:
        raise RuntimeError("Cannot use builtins, '__' or ';'. Sorry.")
    return


def _get_arguments(*args, **kwargs):
    return args, kwargs


def _real_or_int(**kwargs):
    return Integer if kwargs.get('discrete', False) else Real


class DimensionBuilder(object):
    """Create `Dimension` objects using a name for it and an string expression
    which encodes prior and dimension information.

    Basically, one must provide a string like a function call to a method that
    has the name of a distribution, .e.g. ``alpha``, and then provide settings
    about that distributions and information about the `Dimension`, if it
    cannot be inferred. One example for the latter case is:

    ``uniform(-3, 5)`` will return a `Real` dimension, while
    ``uniform(-3, 5, discrete=True)`` will return an `Integer` dimension.

    Sometimes there is also a separate name for the same distribution in integers,
    for the 'uniform' example:

    ``randint(-3, 5)`` will return a uniform `Integer` dimension.

    For categorical dimensions, one can use either ``enum`` or ``random`` name.
    ``random`` however can be used for uniform reals or integers as well.

    Most names are taken from instances contained in `scipy.stats.distributions`.
    So, if the distribution you are searching for is there, then `DimensionBuilder`
    can build one dimension with that prior!

    Examples
    --------
    >>> dimbuilder = DimensionBuilder()
    >>> dimbuilder.build('learning_rate', 'loguniform(0.001, 1, shape=10)')
    Real(name=learning_rate, prior={reciprocal: (0.001, 1), {}}, shape=(10,))
    >>> dimbuilder.build('something_else', 'poisson(mu=3)')
    Integer(name=something_else, prior={poisson: (), {'mu': 3}}, shape=())
    >>> dim = dimbuilder.build('other2', 'random(-5, 2)')
    >>> dim
    Real(name=other2, prior={uniform: (-5, 7), {}}, shape=())
    >>> dim.interval()
    (-5.0, 2.0)

    """

    def __init__(self):
        """Init of `DimensionBuilder`."""
        self.name = None

    def enum(self, *args, **kwargs):
        """Create a `Categorical` dimension."""
        name = self.name
        try:
            if isinstance(args[0], (dict, list)):
                return Categorical(name, *args, **kwargs)
        except IndexError as exc:
            six.raise_from(TypeError("Parameter '{}': "
                                     "Expected argument with categories.".format(name)),
                           exc)

        return Categorical(name, args, **kwargs)

    def random(self, *args, **kwargs):
        """Create `Real` or `Integer` uniform, or `Categorical`."""
        name = self.name
        for arg in args:
            if not isinstance(arg, numbers.Number):
                return self.enum(*args, **kwargs)

        if len(args) > 2:
            # Too many stuff => Categorical
            return self.enum(*args, **kwargs)
        elif len(args) == 2:
            # Change that .@#$% scipy convention for uniform.
            # First is low, second is high.
            klass = _real_or_int(**kwargs)
            return klass(name, 'uniform', args[0], args[1] - args[0], **kwargs)

        # ``len(args) < 2``
        klass = _real_or_int(**kwargs)
        return klass(name, 'uniform', *args, **kwargs)

    def gaussian(self, *args, **kwargs):
        """Synonym for `scipy.stats.distributions.norm`."""
        return self.normal(*args, **kwargs)

    def normal(self, *args, **kwargs):
        """Another synonym for `scipy.stats.distributions.norm`."""
        name = self.name
        klass = _real_or_int(**kwargs)
        return klass(name, 'norm', *args, **kwargs)

    def loguniform(self, *args, **kwargs):
        """Return a `Dimension` object with
        `scipy.stats.distributions.reciprocal` prior distribution.
        """
        name = self.name
        klass = _real_or_int(**kwargs)
        return klass(name, 'reciprocal', *args, **kwargs)

    def _build(self, name, expression):
        """Build a `Dimension` object using a string as its `name` and another
        string, `expression`, from configuration as a "function" to create it.
        """
        self.name = name
        _check_expr_to_eval(expression)
        prior, arg_string = re.findall(r'([a-z][a-z0-9_]*)\((.*)\)', expression)[0]
        try:
            dimension = eval("self." + expression, {'__builtins__': {}},
                             {'self': self})
            return dimension
        except AttributeError:
            pass

        # If not found in the methods of `DimensionBuilder`.
        # try to see if it is legit scipy stuff and call a `Dimension`
        # appropriately.
        args, kwargs = eval("_get_arguments(" + arg_string + ")",
                            {'__builtins__': {}},
                            {'_get_arguments': _get_arguments})

        if hasattr(sp_dists._continuous_distns, prior):
            klass = _real_or_int(**kwargs)
        elif hasattr(sp_dists._discrete_distns, prior):
            klass = Integer
        else:
            raise TypeError("Parameter '{0}': "
                            "'{1}' does not correspond to a supported distribution.".format(
                                name, prior))
        dimension = klass(name, prior, *args, **kwargs)
        return dimension

    def build(self, name, expression):
        """Check `DimensionBuilder._build` for documentation.

        .. note:: Warm-up: Fail early if arguments make object not usable.

        """
        try:
            dimension = self._build(name, expression)
        except ValueError as exc:
            six.raise_from(TypeError(
                "Parameter '{}': Incorrect arguments.".format(name)), exc)
        except IndexError as exc:
            six.raise_from(
                TypeError(
                    "Parameter '{0}': Please provide a valid form for prior:\n"
                    "'distribution(*args, **kwargs)'\nProvided: '{1}'".format(name, expression)),
                exc)

        try:
            dimension.sample()
            dimension.interval()
        except TypeError as exc:
            six.raise_from(TypeError(
                "Parameter '{0}': Incorrect arguments for distribution '{1}'.\n"
                "Scipy Docs::\n\n{2}".format(name,
                                             dimension._prior_name,
                                             dimension.prior.__doc__)), exc)
        except ValueError as exc:
            six.raise_from(TypeError(
                "Parameter '{0}': Incorrect arguments.".format(name)), exc)

        return dimension


def build():
    """Create a definition of the problem's search space, using information
    from the user's script configuration and arguments.

    """
    pass
