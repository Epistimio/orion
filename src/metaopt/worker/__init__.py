# -*- coding: utf-8 -*-
"""
:mod:`metaopt.worker` -- Coordination of the hyperoptimization procedure
========================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Executes hyperoptimization steps and runs training experiment
      with hyperparameter values proposed.

A worker is delegated with various management and functional responsibilities:

   1. It will first try to register to pool of workers, specified by the user
      with a unique string ID (e.g. experiment's name).
   2. It receives write and read access to database, in order to log experiment
      stats, estimated hyperparameter values and validation metrics.
   3. In each hyperoptimization step, it samples an estimation of
      hyperparameters to be used and executes the user's training script with
      these as inputs.
   4. It decides to stop after a finite number of hypersteps and communicates
      with daemon for further instructions.

"""

