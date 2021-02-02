# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
Create Space objects from configuration
=======================================

Functions which build ``Dimension`` and ``Space`` objects for defining problem's search space.

Replace actual hyperparam values in your script's config files or cmd
arguments with orion's keywords for declaring hyperparameter types
to be optimized.

Motivation for this way of orion's configuration is to achieve as
minimal intrusion to user's workflow as possible by:

   * Offering to user the choice to keep the original way of passing
     hyperparameters to their script, be it through some **config file
     type** (e.g. yaml, json, ini, etc) or through **command line
     arguments**.

   * Instead of passing the actual hyperparameter values, use one of
     the characteristic keywords, names enlisted in :scipy.stats:`distributions`
     or :class:`orion.core.io.space_builder.DimensionBuilder`,
     to describe distributions and declare the hyperparameters
     to be optimized. So that a possible command line argument
     like ``-lrate0=0.1`` becomes ``-lrate0~'uniform(-3, 1)'``.

.. note::
   Use ``~`` instead of ``=`` to denote that a variable "draws from"
   a distribution. We support limited Python syntax for describing distributions.

   * Module will also use the script's provided input file/args as a
     template to fill an appropriate input with proposed values for the
     script's execution in each hyperiteration.

"""
import logging
import re
from collections import OrderedDict

from scipy.stats import distributions as sp_dists

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space
from orion.core.utils.flatten import flatten

log = logging.getLogger(__name__)


def _check_expr_to_eval(expr):
    if "__" in expr or ";" in expr:
        raise RuntimeError("Cannot use builtins, '__' or ';'. Sorry.")
    return


def _get_arguments(*args, **kwargs):
    return args, kwargs


def _real_or_int(kwargs):
    return Integer if kwargs.pop("discrete", False) else Real


def replace_key_in_order(odict, key_prev, key_after):
    """Replace ``key_prev`` of ``OrderedDict`` ``odict`` with ``key_after``,
    while leaving its value and the rest of the dictionary intact and in the
    same order.
    """
    tmp = OrderedDict()
    for k, v in odict.items():
        if k == key_prev:
            tmp[key_after] = v
        else:
            tmp[k] = v
    return tmp


def _should_not_be_built(expression):
    return expression.startswith("-") or expression.startswith(">")


def _remove_marker(expression, marker="+"):
    return (
        expression.replace(marker, "", 1)
        if expression.startswith(marker)
        else expression
    )


class DimensionBuilder(object):
    """Create `Dimension` objects using a name for it and an string expression
    which encodes prior and dimension information.

    Basically, one must provide a string like a function call to a method that
    has the name of a distribution, .e.g. ``alpha``, and then provide settings
    about that distributions and information about the `Dimension`, if it
    cannot be inferred. One example for the latter case is:

    ``uniform(-3, 5)`` will return a :class:`orion.algo.space.Real` dimension, while
    ``uniform(-3, 5, discrete=True)`` will return an :class:`orion.algo.space.Integer` dimension.

    Sometimes there is also a separate name for the same distribution in integers,
    for the 'uniform' example:

    ``randint(-3, 5)`` will return a uniform :class:`orion.algo.space.Integer` dimension.

    For categorical dimensions, one can use either ``enum`` or ``random`` name.
    ``random`` however can be used for uniform reals or integers as well.

    Most names are taken from instances contained in :scipy.stats:`distributions`.
    So, if the distribution you are searching for is there, then `DimensionBuilder`
    can build one dimension with that prior!

    Examples
    --------
    >>> dimbuilder = DimensionBuilder()
    >>> dimbuilder.build('learning_rate', 'loguniform(0.001, 1, shape=10)')
    Real(name=learning_rate, prior={reciprocal: (0.001, 1), {}}, shape=(10,))
    >>> dimbuilder.build('something_else', 'poisson(mu=3)')
    Integer(name=something_else, prior={poisson: (), {'mu': 3}}, shape=())
    >>> dim = dimbuilder.build('other2', 'uniform(-5, 2)')
    >>> dim
    Real(name=other2, prior={uniform: (-5, 7), {}}, shape=())
    >>> dim.interval()
    (-5.0, 2.0)

    """

    def __init__(self):
        """Init of `DimensionBuilder`."""
        self.name = None

    def choices(self, *args, **kwargs):
        """Create a :class:`orion.algo.space.Categorical` dimension."""
        name = self.name
        try:
            if isinstance(args[0], (dict, list)):
                return Categorical(name, *args, **kwargs)
        except IndexError as exc:
            raise TypeError(
                "Parameter '{}': " "Expected argument with categories.".format(name)
            ) from exc

        return Categorical(name, args, **kwargs)

    def fidelity(self, *args, **kwargs):
        """Create a :class:`orion.algo.space.Fidelity` dimension."""
        name = self.name
        return Fidelity(name, *args, **kwargs)

    def uniform(self, *args, **kwargs):
        """Create an :class:`orion.algo.space.Integer` or :class:`orion.algo.space.Real` uniformly
        distributed dimension.

        .. note:: Changes scipy convention for uniform's arguments. In scipy,
           ``uniform(a, b)`` means uniform in the interval [a, a+b). Here, it
           means uniform in the interval [a, b].

        """
        name = self.name
        klass = _real_or_int(kwargs)
        if len(args) == 2:
            return klass(name, "uniform", args[0], args[1] - args[0], **kwargs)
        return klass(name, "uniform", *args, **kwargs)

    def randint(self, *args, **kwargs):
        """Create an :class:`orion.algo.space.Integer` or :class:`orion.algo.space.Real` uniformly
        distributed dimension.

        .. note:: Changes scipy convention for uniform's arguments. In scipy,
           ``uniform(a, b)`` means uniform in the interval [a, a+b). Here, it
           means uniform in the interval [a, b].

        """
        raise NotImplementedError(
            "`randint` is not supported. Use uniform(discrete=True) instead."
        )

    def gaussian(self, *args, **kwargs):
        """Synonym for :scipy.stats:`distributions.norm`."""
        return self.normal(*args, **kwargs)

    def normal(self, *args, **kwargs):
        """Another synonym for :scipy.stats:`distributions.norm`."""
        name = self.name
        klass = _real_or_int(kwargs)
        return klass(name, "norm", *args, **kwargs)

    def loguniform(self, *args, **kwargs):
        """Return a `Dimension` object with
        :scipy.stats:`distributions.reciprocal` prior distribution.
        """
        name = self.name
        klass = _real_or_int(kwargs)
        return klass(name, "reciprocal", *args, **kwargs)

    def _build(self, name, expression):
        """Build a `Dimension` object using a string as its `name` and another
        string, `expression`, from configuration as a "function" to create it.
        """
        self.name = name
        _check_expr_to_eval(expression)

        prior, arg_string = re.findall(r"([a-z][a-z0-9_]*)\((.*)\)", expression)[0]
        globals_ = {"__builtins__": {}}
        try:
            dimension = eval("self." + expression, globals_, {"self": self})

            return dimension
        except AttributeError:
            pass

        # If not found in the methods of `DimensionBuilder`.
        # try to see if it is legit scipy stuff and call a `Dimension`
        # appropriately.
        args, kwargs = eval(
            "_get_arguments(" + arg_string + ")",
            globals_,
            {"_get_arguments": _get_arguments},
        )

        if hasattr(sp_dists._continuous_distns, prior):
            klass = _real_or_int(kwargs)
        elif hasattr(sp_dists._discrete_distns, prior):
            klass = Integer
        else:
            raise TypeError(
                "Parameter '{0}': "
                "'{1}' does not correspond to a supported distribution.".format(
                    name, prior
                )
            )
        dimension = klass(name, prior, *args, **kwargs)

        return dimension

    def build(self, name, expression):
        """Check ``DimensionBuilder._build`` for documentation.

        .. note:: Warm-up: Fail early if arguments make object not usable.

        """
        try:
            dimension = self._build(name, expression)
        except ValueError as exc:
            raise TypeError(
                "Parameter '{}': Incorrect arguments.".format(name)
            ) from exc
        except IndexError as exc:
            error_msg = (
                "Parameter '{0}': Please provide a valid form for prior:\n"
                "'distribution(*args, **kwargs)'\n"
                "Provided: '{1}'".format(name, expression)
            )
            raise TypeError(error_msg) from exc

        try:
            dimension.sample()
        except TypeError as exc:
            error_msg = (
                "Parameter '{0}': Incorrect arguments for distribution '{1}'.\n"
                "Scipy Docs::\n\n{2}".format(
                    name, dimension._prior_name, dimension.prior.__doc__
                )
            )
            raise TypeError(error_msg) from exc
        except ValueError as exc:
            raise TypeError(
                "Parameter '{0}': Incorrect arguments.".format(name)
            ) from exc

        return dimension


class SpaceBuilder(object):
    """Build a :class:`orion.algo.space.Space` object form user's configuration."""

    def __init__(self):
        self.dimbuilder = DimensionBuilder()
        self.space = None

        self.commands_tmpl = None

        self.converter = None
        self.parser = None

    def build(self, configuration):
        """Create a definition of the problem's search space.

        Using information from the user's script configuration (if provided) and the
        command line arguments, will create a :class:`orion.algo.space.Space` object defining the
        problem's search space.

        Parameters
        ----------
        configuration: OrderedDict
            An OrderedDict containing the name and the expression of the parameters.

        Returns
        -------
        :class:`orion.algo.space.Space`
            The problem's search space definition.

        """
        self.space = Space()
        for namespace, expression in flatten(configuration).items():
            if _should_not_be_built(expression):
                continue

            expression = _remove_marker(expression)
            dimension = self.dimbuilder.build(namespace, expression)

            try:
                self.space.register(dimension)
            except ValueError as exc:
                error_msg = "Conflict for name '{}' in parameters".format(namespace)
                raise ValueError(error_msg) from exc

        return self.space

    def build_to(self, config_path, trial, experiment=None):
        """Create the configuration for the user's script.

        Using the configuration parser, create the commandline associated with the
        user's script while replacing the correct instances of parameter distributions by
        their actual values. If needed, the parser will also create a configuration file.

        Parameters
        ----------
        config_path: str
            Path in which the configuration file instance will be created.
        trial: `orion.core.worker.trial.Trial`
            Object with concrete parameter values for the defined :class:`orion.algo.space.Space`.
        experiment: :class:`orion.core.worker.experiment.Experiment`, optional
            Object with information related to the current experiment.

        Returns
        -------
        list
            The commandline arguments that must be given to script for execution.

        """
        return self.parser.format(config_path, trial, experiment)
