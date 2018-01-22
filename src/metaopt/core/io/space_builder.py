# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`metaopt.core.io.space_builder` -- Create Space objects from configuration
===============================================================================

.. module:: space_builder
   :platform: Unix
   :synopsis: Functions which build `Dimension` and `Space` objects for
      defining problem's search space.

Replace actual hyperparam values in your script's config files or cmd
arguments with metaopt's keywords for declaring hyperparameter types
to be optimized.

Motivation for this way of metaopt's configuration is to achieve as
minimal intrusion to user's workflow as possible by:

   * Offering to user the choice to keep the original way of passing
   hyperparameters to their script, be it through some **config file
   type** (e.g. yaml, json, ini, etc) or through **command line
   arguments**.

   * Instead of passing the actual hyperparameter values, use one of
   the characteristic keywords, names enlisted in `scipy.stats.distributions`
   or `metaopt.core.io.space_builder.DimensionBuilder`,
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

import collections
import copy
import logging
import numbers
import os
import re
import sys

from scipy.stats import distributions as sp_dists
import six

from metaopt.algo.space import (Categorical, Integer, Real, Space)
from metaopt.core.io.convert import infer_converter_from_file_type


log = logging.getLogger(__name__)


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
        if sys.version_info[0] == 3:  # if Python3
            globals_ = {'__builtins__': {}}
        else:  # if Python2
            # Try False = True somewhere in Python2, you will see...
            globals_ = {'__builtins__': {'True': True, 'False': False}}
        try:
            dimension = eval("self." + expression, globals_, {'self': self})
            return dimension
        except AttributeError:
            pass

        # If not found in the methods of `DimensionBuilder`.
        # try to see if it is legit scipy stuff and call a `Dimension`
        # appropriately.
        args, kwargs = eval("_get_arguments(" + arg_string + ")",
                            globals_,
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


class SpaceBuilder(object):
    """Build a `Space` object form user's configuration."""

    USERCONFIG_OPTION = '--config='
    USERCONFIG_KEYWORD = 'mopt~'
    USERARGS_SEARCH = r'\W*([a-zA-Z0-9_-]+)~(.*)'
    USERARGS_TMPL = r'(.*)~(.*)'

    def __init__(self):
        """Initialize a `SpaceBuilder`."""
        self.userconfig = None
        self.is_userconfig_an_option = None
        self.userargs_tmpl = None
        self.userconfig_tmpl = None
        self.dimbuilder = DimensionBuilder()
        self.space = None
        self.converter = None

    def build_from(self, cmd_args):
        """Create a definition of the problem's search space, using information
        from the user's script configuration (if provided) and command line arguments.

        :param cmd_args: A list of command line arguments provided for the user's script.

        .. note:: A template configuration file complementing user's script can be
           provided either by explicitly using the prefix '--config=' or by being the
           first positional argument.

        """
        self.userargs_tmpl = None
        self.userconfig_tmpl = None
        self.space = Space()

        self.userconfig, self.is_userconfig_an_option = self._build_from_args(cmd_args)

        if self.userconfig:
            self._build_from_config(self.userconfig)

        log.debug("Configuration and command line arguments were parsed and "
                  "a `Space` object was built successfully:\n%s", self.space)

        return self.space

    def _build_from_config(self, config_path):
        self.converter = infer_converter_from_file_type(config_path)
        self.userconfig_tmpl = self.converter.parse(config_path)

        stack = collections.deque()
        stack.append(('', self.userconfig_tmpl))
        while True:
            try:
                namespace, stuff = stack.pop()
            except IndexError:
                break
            if isinstance(stuff, dict):
                for k, v in six.iteritems(stuff):
                    stack.append(('/'.join([namespace, str(k)]), v))
            elif isinstance(stuff, list):
                for position, thing in enumerate(stuff):
                    stack.append(('/'.join([namespace, str(position)]), thing))
            elif isinstance(stuff, six.string_types):
                if stuff.startswith(self.USERCONFIG_KEYWORD):
                    dimension = self.dimbuilder.build(namespace,
                                                      stuff[len(self.USERCONFIG_KEYWORD):])
                    try:
                        self.space.register(dimension)
                    except ValueError as exc:
                        six.raise_from(
                            ValueError(
                                "Conflict for name '%s' in script configuration and arguments.",
                                namespace),
                            exc)

    def _build_from_args(self, cmd_args):
        userconfig = None
        is_userconfig_an_option = None
        self.userargs_tmpl = collections.defaultdict(list)
        args_pattern = re.compile(self.USERARGS_SEARCH)
        args_prefix_pattern = re.compile(self.USERARGS_TMPL)

        for arg in cmd_args:
            found = args_pattern.findall(arg)
            if len(found) != 1:
                if arg.startswith(self.USERCONFIG_OPTION):
                    if not userconfig:
                        userconfig = arg[len(self.USERCONFIG_OPTION):]
                        is_userconfig_an_option = True
                    else:
                        raise ValueError(
                            "Already found one configuration file in: %s",
                            userconfig
                            )
                else:
                    self.userargs_tmpl[None].append(arg)
                continue

            name, expression = found[0]
            namespace = '/' + name
            dimension = self.dimbuilder.build(namespace, expression)
            self.space.register(dimension)

            found = args_prefix_pattern.findall(arg)
            assert len(found) == 1 and found[0][1] == expression, "Parsing prefix problem."
            self.userargs_tmpl[namespace] = found[0][0] + '='

        if not userconfig and self.userargs_tmpl:  # try the first positional argument
            if os.path.isfile(self.userargs_tmpl[None][0]):
                userconfig = self.userargs_tmpl[None].pop(0)
                is_userconfig_an_option = False

        return userconfig, is_userconfig_an_option

    def build_to(self, config_path, trial):
        """Use templates saved from `build_from` to generate a config file (if needed)
        and command line arguments to correspond to specific parameter selections.

        :param config_path: Path in which the configuration file instance
           will be created.
        :param trial: A `metaopt.core.worker.trial.Trial` object with concrete
           parameter values for the defined `Space`.

        """
        if self.userconfig:
            self._build_to_config(config_path, trial)
        return self._build_to_args(config_path, trial)

    def _build_to_config(self, config_path, trial):
        config_instance = copy.deepcopy(self.userconfig_tmpl)

        for param in trial.params:
            stuff = config_instance
            path = param.name.split('/')
            for key in path[1:]:
                # Parameter name may correspond to stuff in cmd args
                if isinstance(stuff, list):
                    key = int(key)
                    try:
                        stuff[key]
                    except IndexError:
                        break
                else:  # isinstance(stuff, dict):
                    if key not in stuff:
                        break

                if isinstance(stuff[key], six.string_types):
                    stuff[key] = param.value
                else:
                    stuff = stuff[key]

        self.converter.generate(config_path, config_instance)

    def _build_to_args(self, config_path, trial):
        cmd_args = []

        if self.userconfig:
            if self.is_userconfig_an_option:
                cmd_args.append(self.USERCONFIG_OPTION + config_path)
            else:
                cmd_args.append(config_path)

        cmd_args.extend(self.userargs_tmpl[None])

        for param in trial.params:
            if param.name not in self.userargs_tmpl:
                continue
            prefix = self.userargs_tmpl[param.name]
            cmd_args.append(prefix + str(param.value))

        return cmd_args
