# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`orion.core.io.space_builder` -- Create Space objects from configuration
=============================================================================

.. module:: space_builder
   :platform: Unix
   :synopsis: Functions which build `Dimension` and `Space` objects for
      defining problem's search space.

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
     the characteristic keywords, names enlisted in `scipy.stats.distributions`
     or `orion.core.io.space_builder.DimensionBuilder`,
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
import os
import re

from scipy.stats import distributions as sp_dists

from orion.algo.space import (Categorical, Fidelity, Integer, Real, Space)
from orion.core.io.convert import infer_converter_from_file_type

log = logging.getLogger(__name__)


def _check_expr_to_eval(expr):
    if '__' in expr or ';' in expr:
        raise RuntimeError("Cannot use builtins, '__' or ';'. Sorry.")
    return


def _get_arguments(*args, **kwargs):
    return args, kwargs


def _real_or_int(kwargs):
    return Integer if kwargs.pop('discrete', False) else Real


def replace_key_in_order(odict, key_prev, key_after):
    """Replace `key_prev` of `OrderedDict` `odict` with `key_after`,
    while leaving its value and the rest of the dictionary intact and in the
    same order.
    """
    tmp = collections.OrderedDict()
    for k, v in odict.items():
        if k == key_prev:
            tmp[key_after] = v
        else:
            tmp[k] = v
    return tmp


def _should_not_be_built(expression):
    return expression.startswith('-') or expression.startswith('>')


def _remove_marker(expression, marker='+'):
    return expression.replace(marker, '', 1) if expression.startswith(marker) else expression


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

    def choices(self, *args, **kwargs):
        """Create a `Categorical` dimension."""
        name = self.name
        try:
            if isinstance(args[0], (dict, list)):
                return Categorical(name, *args, **kwargs)
        except IndexError as exc:
            raise TypeError("Parameter '{}': "
                            "Expected argument with categories.".format(name)) from exc

        return Categorical(name, args, **kwargs)

    def fidelity(self):
        """Create a `Fidelity` dimension."""
        name = self.name
        return Fidelity(name)

    def uniform(self, *args, **kwargs):
        """Create an `Integer` or `Real` uniformly distributed dimension.

        .. note:: Changes scipy convention for uniform's arguments. In scipy,
           ``uniform(a, b)`` means uniform in the interval [a, a+b). Here, it
           means uniform in the interval [a, b).

        """
        name = self.name
        klass = _real_or_int(kwargs)
        if len(args) == 2:
            return klass(name, 'uniform', args[0], args[1] - args[0], **kwargs)
        return klass(name, 'uniform', *args, **kwargs)

    def gaussian(self, *args, **kwargs):
        """Synonym for `scipy.stats.distributions.norm`."""
        return self.normal(*args, **kwargs)

    def normal(self, *args, **kwargs):
        """Another synonym for `scipy.stats.distributions.norm`."""
        name = self.name
        klass = _real_or_int(kwargs)
        return klass(name, 'norm', *args, **kwargs)

    def loguniform(self, *args, **kwargs):
        """Return a `Dimension` object with
        `scipy.stats.distributions.reciprocal` prior distribution.
        """
        name = self.name
        klass = _real_or_int(kwargs)
        return klass(name, 'reciprocal', *args, **kwargs)

    def _build(self, name, expression):
        """Build a `Dimension` object using a string as its `name` and another
        string, `expression`, from configuration as a "function" to create it.
        """
        self.name = name
        _check_expr_to_eval(expression)

        prior, arg_string = re.findall(r'([a-z][a-z0-9_]*)\((.*)\)', expression)[0]
        globals_ = {'__builtins__': {}}
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
            klass = _real_or_int(kwargs)
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
            raise TypeError("Parameter '{}': Incorrect arguments.".format(name)) from exc
        except IndexError as exc:
            error_msg = "Parameter '{0}': Please provide a valid form for prior:\n"\
                        "'distribution(*args, **kwargs)'\n"\
                        "Provided: '{1}'".format(name, expression)
            raise TypeError(error_msg) from exc

        try:
            dimension.sample()
        except TypeError as exc:
            error_msg = "Parameter '{0}': Incorrect arguments for distribution '{1}'.\n"\
                        "Scipy Docs::\n\n{2}".format(name,
                                                     dimension._prior_name,
                                                     dimension.prior.__doc__)
            raise TypeError(error_msg) from exc
        except ValueError as exc:
            raise TypeError("Parameter '{0}': Incorrect arguments.".format(name)) from exc

        return dimension


class SpaceBuilder(object):
    """Build a `Space` object form user's configuration."""

    # TODO Expose these 4 USER oriented goodfellows to a orion configuration file :)
    USERCONFIG_KEYWORD = 'orion~'
    USERARGS_TMPL = r'(.*)~([\+\-\>]?.*)'
    USERARGS_NAMESPACE = r'\W*([a-zA-Z0-9_-]+)'
    USERARGS_CONFIG = '--config='

    EXPOSED_PROPERTIES = ['trial.hash_name', 'trial.full_name', 'exp.name']

    def __init__(self):
        """Initialize a `SpaceBuilder`."""
        self.userconfig = None
        self.userargs_tmpl = None
        self.userconfig_tmpl = None
        self.userconfig_expressions = collections.OrderedDict()
        self.userconfig_nameless = collections.OrderedDict()
        self.positional_args_count = 0

        self.dimbuilder = DimensionBuilder()
        self.space = None

        self.commands_tmpl = None

        self.converter = None

    def build_from(self, cmd_args):
        """Create a definition of the problem's search space, using information
        from the user's script configuration (if provided) and command line arguments.

        This method is also responsible for parsing semantics which allow
        information from the `Experiment` or a `Trial` object to be passed
        to user's script. For example, a command line option like
        ``--name~trial.hash_name`` will be parsed to mean that a unique hash
        identifier of the trial that defines an execution shall be passed to
        the option ``--name``. Usage of these properties are not obligatory,
        but it helps integrating with solutions which help experiment
        reproducibility and resumability. See :attr:`EXPOSED_PROPERTIES` to
        check which properties are supported.

        :param cmd_args: A list of command line arguments provided for the user's script.

        :rtype: `orion.algo.space.Space`

        .. note:: A template configuration file complementing user's script can be
           provided either by explicitly using the prefix '--config=' or by being the
           first positional argument.

        """
        self.userargs_tmpl = None
        self.userconfig_tmpl = None
        self.userconfig_expressions = collections.OrderedDict()
        self.userconfig_nameless = collections.OrderedDict()

        self.commands_tmpl = None
        self.space = Space()

        self.userconfig = self._build_from_args(cmd_args)

        if self.userconfig:
            self._build_from_config(self.userconfig)

        log.debug("Configuration and command line arguments were parsed and "
                  "a `Space` object was built successfully:\n%s", self.space)

        return self.space

    def _build_from_config(self, config_path):
        self.converter = infer_converter_from_file_type(config_path,
                                                        default_keyword=self.USERCONFIG_KEYWORD)
        self.userconfig_tmpl = self.converter.parse(config_path)
        self.userconfig_expressions = collections.OrderedDict()
        self.userconfig_nameless = collections.OrderedDict()

        stack = collections.deque()
        stack.append(('', self.userconfig_tmpl))
        while True:
            try:
                namespace, stuff = stack.pop()
            except IndexError:
                break
            if isinstance(stuff, dict):
                for k, v in stuff.items():
                    stack.append(('/'.join([namespace, str(k)]), v))
            elif isinstance(stuff, list):
                for position, thing in enumerate(stuff):
                    stack.append(('/'.join([namespace, str(position)]), thing))
            elif isinstance(stuff, str):
                if stuff.startswith(self.USERCONFIG_KEYWORD):
                    expression = stuff[len(self.USERCONFIG_KEYWORD):]

                    # Store the expression before it is modified for the dimension builder
                    self.userconfig_expressions[namespace] = (
                        self.USERCONFIG_KEYWORD[-1] + expression)

                    if _should_not_be_built(expression):
                        break

                    expression = _remove_marker(expression)

                    dimension = self.dimbuilder.build(namespace, expression)
                    try:
                        self.space.register(dimension)
                    except ValueError as exc:
                        error_msg = "Conflict for name '{}' in script configuration "\
                                    "and arguments.".format(namespace)
                        raise ValueError(error_msg) from exc
                else:
                    log.info("Nameless '%s: %s' will not define a dimension.", namespace, stuff)
                    self.userconfig_nameless[namespace] = stuff

    def _build_from_args(self, cmd_args):
        """Build templates from arguments found in the original cli.

        Rules for namespacing:
           1. Prefix ``'/'`` is given to parameter dimensions.
           2. Prefix ``'$'`` is given to substitutions from exposed properties.
           3. Prefix ``'_'`` is given to arguments which do not interact with.
              Suffix is a unique integer.
           4. ``'config'`` is given to user's script configuration file template

        User script configuration file argument is treated specially, trying to
        recognise a ``--config`` option or by checking the first positional
        argument, if not found elsewhere. Only one is allowed.

        .. note:: Positional arguments cannot define a parameter
           dimension, because no **meaningful** name can be assigned.

        .. note:: Templates preserve given argument order.

        """
        userconfig = None
        self.userargs_tmpl = collections.OrderedDict()
        self.commands_tmpl = collections.OrderedDict()
        args_pattern = re.compile(self.USERARGS_TMPL)
        args_namespace_pattern = re.compile(self.USERARGS_NAMESPACE)

        self.positional_args_count = 0

        def get_next_pos_ns():
            """Generate next namespace for a positional argument."""
            ns = str(self.positional_args_count)
            self.positional_args_count += 1
            return ns

        args_value = collections.OrderedDict()
        i = 0
        while i < len(cmd_args):
            args_value[cmd_args[i]] = None

            if cmd_args[i].startswith('-') and i < len(cmd_args) - 1:
                if not cmd_args[i + 1].startswith('-'):
                    args_value[cmd_args[i]] = cmd_args[i + 1]
                    i += 1
            i += 1

        for arg, value in args_value.items():
            if value is not None:
                arg = arg + "=" + value

            found = args_pattern.findall(arg)
            if len(found) != 1:
                if arg.startswith(self.USERARGS_CONFIG):
                    if not userconfig:
                        userconfig = arg[len(self.USERARGS_CONFIG):]
                        self.userargs_tmpl['config'] = self.USERARGS_CONFIG
                    else:
                        raise ValueError(
                            "Already found one configuration file in: %s" %
                            userconfig
                            )
                else:
                    self.userargs_tmpl['_' + get_next_pos_ns()] = arg
                continue

            prefix, expression = found[0]
            ns_search = args_namespace_pattern.findall(prefix)
            if expression in self.EXPOSED_PROPERTIES:
                # It's a experiment/trial management command
                namespace = '$'
                namespace += ns_search[0] if ns_search else get_next_pos_ns()
                objname, attrname = expression.split('.')
                self.commands_tmpl[namespace] = (objname, attrname)
                self.userargs_tmpl[namespace] = prefix + '=' if ns_search else ''
            elif not ns_search or not expression or expression[0] == '/':
                # This branch targets the nameless and the ones that use `~`
                # as the home directory in a shell path
                # If it's nameless (positional) it cannot be a dimension
                log.info("Nameless argument '%s' will not define a dimension.", arg)
                self.userargs_tmpl['_' + get_next_pos_ns()] = arg
            elif not _should_not_be_built(expression):
                # Otherwise it's a dimension; ikr
                namespace = '/' + ns_search[0]

                expression = _remove_marker(expression)

                dimension = self.dimbuilder.build(namespace, expression)

                self.space.register(dimension)
                self.userargs_tmpl[namespace] = prefix + '='

        # Risky assumption about the first non-interacting argument being a
        # user's script configuration file. May be dropped in the future.
        # Quite surely if TODO @221 is implemented because user can specify
        # how their configuration is supposed to be parsed, so ciao.
        # For now issue a warning if this is going to happen.
        if not userconfig and '_0' in self.userargs_tmpl:
            if os.path.isfile(self.userargs_tmpl['_0']):
                userconfig = self.userargs_tmpl['_0']
                log.info("Using '%s' as path to user script's configuration file!", userconfig)
                # '_0' is 'config' now, replace key in the correct order
                self.userargs_tmpl = replace_key_in_order(self.userargs_tmpl, '_0', 'config')
                self.userargs_tmpl['config'] = ''

        return userconfig

    def build_to(self, config_path, trial, experiment=None):
        """Use templates saved from `build_from` to generate a config file (if needed)
        and command line arguments to correspond to specific parameter selections.

        :param config_path: Path in which the configuration file instance
           will be created.
        :param trial: A `orion.core.worker.trial.Trial` object with concrete
           parameter values for the defined `Space`. It may be used to retrieve
           managerial information from it.
        :param experiment: An `orion.core.worker.experiment.Experiment` object.
           It may be used to retrieve managerial information from it. See
           :attr:`EXPOSED_PROPERTIES`.

        :returns: A list with the command line arguments that must be given to
           script's execution.

        """
        if self.userconfig:
            self._build_to_config(config_path, trial)
        return self._build_to_args(config_path, trial, experiment)

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

                if isinstance(stuff[key], str):
                    stuff[key] = param.value
                else:
                    stuff = stuff[key]

        self.converter.generate(config_path, config_instance)

    def _build_to_args(self, config_path, trial, experiment=None):
        cmd_args = []
        # objects whose properties can be fetched to fill a cli argument
        exposed_objects = {'trial': trial, 'exp': experiment}
        param_names = [trial.params[i].name for i in range(len(trial.params))]

        for namespace, prefix in self.userargs_tmpl.items():
            if namespace == 'config':
                cmd_args.append(prefix + config_path)
            elif namespace.startswith('/'):
                param_idx = param_names.index(namespace)
                cmd_args.append(prefix + str(trial.params[param_idx].value))
            elif namespace.startswith('$'):
                objname, attrname = self.commands_tmpl[namespace]
                item = getattr(exposed_objects[objname], attrname)
                cmd_args.append(prefix + item)
            elif namespace.startswith('_'):
                cmd_args.append(prefix)

        return cmd_args
