# -*- coding: utf-8 -*-
"""
:mod:`metaopt.worker.convert` -- Parse and generate user script's configuration
===============================================================================

.. module:: convert
   :platform: Unix
   :synopsis: Search for hparam metaopt declarations and generate files or
      command line arguments correspondingly as inputs for user's script in
      each hyperiteration.

Usage:
   Replace actual hyperparam values in your script's config files or cmd
   arguments with :mod:`metaopt`'s keywords for declaring hyperparameter types
   to be optimized.

   Motivation for this way of :mod:`metaopt`'s configuration is to achieve as
   minimal intrusion to user's workflow as possible by:

      * Offering to user the choice to keep the original way of passing
        hyperparameters to their script, be it through some **config file
        type** (e.g. yaml, json, ini, etc) or through **command line
        arguments**.

      * Instead of passing the actual hyperparameter values, use one of
        the characteristic keywords, enlisted in :mod:`metaopt.optim.distribs`,
        to describe distributions and declare the hyperparameters
        to be optimized. So that a possible command line argument
        like `-lrate0=0.1` becomes `-lrate0~'10**uniform(-3,1)'`.

.. note::
   Use `~` instead of `=` to denote that a varible "draws from"
   a distribution. We support Python syntax for describing distributions.

      * Module will also use the script's provided input file/args as a
        template to fill an appropriate input with proposed values for the
        script's execution in each hyperiteration.

"""
