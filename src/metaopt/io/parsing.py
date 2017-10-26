#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import socket
import argparse
import textwrap
import six
from collections import defaultdict

from metaopt import __version__
from metaopt import optim
from metaopt import dynamic

# Define type of arbitrary nested defaultdicts
nesteddict = lambda: defaultdict(nesteddict)

# cmd > config > database > env > default

def mopt_args(description):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(description))
    parser.add_argument('-V', '--version',
                        action='version', version='metaopt ' + __version__)
    # TODO: Also, can be fetched from the moptconfig, but in general cmd>config
    parser.add_argument('-e', '--name',
                        type=str, metavar='ID',
                        help='experiment\'s unique name')
    # TODO: Number of trials?? Add argument put also put a default.
    # TODO: a default/best moptconfig would help
    parser.add_argument('-mc', '--moptconfig',
                        type=argparse.FileType('r'), metavar='path',
                        help='user provided mopt configuration file')
    parser.add_argument('-c', '--userconfig',
                        type=str, metavar='path',
                        help='your script\'s configuration file (yaml, json, ini, ...anything)')
    parser.add_argument('userscript',
                        type=str, metavar='path',
                        help='your experiment\'s script')
    parser.add_argument('userargs',
                        nargs=argparse.REMAINDER, metavar='...',
                        help='command line arguments to your script (if any)')

    args = vars(parser.parse_args())
    expconfig = nesteddict()
    for k, v in six.iteritems(args):
        expconfig[k] = v
    expconfig['version'] = __version__
    return expconfig


# list containing tuples of
# (environmental variable names, configuration keys, default values)
MOPT_DB_ENV_VARS = [
    ('METAOPT_DB_NAME', 'name', 'MetaOpt'),
    ('METAOPT_DB_TYPE', 'type', 'MongoDB'),
    ('METAOPT_DB_ADDRESS', 'address', socket.gethostbyname(socket.gethostname()))
    ]

# TODO Resource from environmental

# dictionary describing lists of environmental tuples (e.g. `MOPT_DB_ENV_VARS`)
# by a 'key' to be used in the experiment's configuration dict
MOPT_ENV_VARS = dict(
    database=MOPT_DB_ENV_VARS
    )


def env_vars(args):
    for signif, evars in six.iteritems(MOPT_ENV_VARS):
        for var_name, key, default_value in evars:
            value = os.getenv(var_name, default_value)
            args[signif][key] = value
    return args


def mopt_config(args):
    """
    Substitutes 'moptconfig' file with the configuration it describes
    inside the experiment's configuration dict, `args`.

    'moptconfig' can describe:
       * 'name': Experiment's name. If you provide a past experiment's name,
         then that experiment will be resumed. This means that its history of
         trials will be reused, along with any configurations logged in the
         database which are not overwritten by current call to `mopt` script.
         (default: <username>_<start datetime>)
       * 'numTrials': Maximum number of trial evaluations to be computed
         (required as a cmd line argument or a moptconfig parameter)
       * 'workers': Number of workers evaluating in parallel asychronously
         (default: 1@default resource). Can be a dict of the form:
         {resource_alias: numWorkers}
       * 'resources': {resource_alias: (entry_address, scheduler)} (optional)

       * 'optimizer': {optimizer module name : method-specific configuration}
       * 'dynamic': {dynamic module name : method-specific configuration}

       .. seealso:: Method-specific configurations reside in `/config`

    """
    moptfile = args.pop('moptconfig')
    assert(moptfile is not None)
    # TODO voici et voila
    # For whatever not in user provided moptconfig, draw from module defaults!
    # XXX Module defaults should be written to an example configuration file
    # automatically as pre-commit hooks!!
    # 1. check whether supplied method{optimizer, dynamic} names exist
    # 2. check whether supplied specific methods/parameters are correct
    return args
