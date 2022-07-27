"""
==========================
Local Parameter Importance
==========================

.. hint:: 

   Conveys a very compact measure of the importance of the different hyperparameters
   to achieve the best objective found so far.

The local parameter importance measures the variance of the results when
varying one hyperparameter and keeping all other fixes [Biedenkapp2018]_.

Given a best set of hyperparameters, we separately build a grid
for each hyperparameter and compute the variance of the results when keeping all other
hyperparameter fixed to their values in the best set. In order to infer these results,
we train a regression model from scikit-learn (by default RandomForestRegressor)
on the trial history of the experiment, and use it to predict the objective.
The ratio of variance for one hyperparameter versus the sum of variances for all hyperparameters
is used as the local parameter importance metric.

.. autofunction:: orion.plotting.base.lpi
    :noindex:

The local parameter importance plot can be executed directly from the ``experiment`` with
``plot.lpi()`` as shown in the example below.

"""
from orion.client import get_experiment

# flake8: noqa: E265

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.lpi()
fig

#%%
# On this plot the x-axis shows the different hyperparameters while the y-axis gives the
# local parameter importance.
# The error bars represent the standard deviation of the LPI based on 10 runs. Remember
# that the LPI requires the training of a regression model. The initial state of this model can
# greatly influence the results. This is why the computation of the LPI is done multiple times
# (10 times by default) using different random seeds. Here is an example
# setting the number of points for the grids, the number of runs and the initial random seed
# for the regression model.

experiment.plot.lpi(n_points=10, n_runs=5, model_kwargs=dict(random_state=1))

#%%
# We can see that the learning rate had a larger impact than the dropout in achieving the best
# objective.
# A search space of only 2 dimensions is easy to analyse visually however so the LPI
# has little additional value in this example. We need to use an example with a larger search space
# to better show to utility of the LPI. We will load results from tutorial
# :ref:`sphx_glr_auto_tutorials_code_2_hyperband_checkpoint.py` for this.

# Load the data for the specified experiment
experiment = get_experiment("hyperband-cifar10", storage=storage)
experiment.plot.lpi()

#%%
# There is a large difference here between the most important hyperparameter (learning rate) and the
# least important one (gamma).
#
# One caveat of LPI is that the variance for each hyperparameters depends on the search space.
# If the prior for one hyperparameter is narrow and fits the region of best values for this
# hyperparameter, then the variance will be low and this hyperparameter will be considered
# non-important. It may be important, but it is not important to optimize it within this narrow
# search space. Another related issue, is that if one hyperparameter have a dramatic effect, it will
# lead to a variance so large that the other hyperparameters will seem unrelevant in comparison.
# This is what we observe here with the learning rate. If we branch from the experiment and define
# a narrowed prior for the learning rate, we will see that it becomes an unimportant
# hyperparameter.
# See documentation on :ref:`EVC system` for more information on branching, or
# :py:func:`orion.client.build_experiment` for information on ``branching`` arguments.
# Original learning rate prior was ``loguniform(1e-5, 0.1)``. We will narrow it to
# ``loguniform(1e-3, 0.1)``.

from orion.client import build_experiment

# Branch from "hyperband-cifar10" with a narrower search space.
experiment = build_experiment(
    "narrow-hyperband-cifar10",
    branching={"branch_from": "hyperband-cifar10"},
    space={
        "epochs": "fidelity(1, 120, base=4)",
        "learning_rate": "loguniform(1e-3, 0.1)",
        "momentum": "uniform(0, 0.9)",
        "weight_decay": "loguniform(1e-10, 1e-2)",
        "gamma": "loguniform(0.97, 1)",
    },
    storage=storage,
)

experiment.plot.lpi()

#%%
# The prior of the learning rate is arguably large, spanning over 3 orders of magnitude
# `(0.001, 0.1)`. Nevertheless, for this problem, most learning rates within this range
# leads to optimal results whenever the other hyperparameters are optimal. What you must remember
# is that defining to narrow search spaces may lead to misleading local parameter importance.
# See :ref:`sphx_glr_auto_examples_plot_4_partial_dependencies.py` for a visualization to verify if
# the search space you defined may be too narrow.
#
# Special cases
# -------------
#
# Logarithmic scale
# ~~~~~~~~~~~~~~~~~
#
# Dimensions with a logarithmic prior :ref:`search-space-prior-loguniform` are linearized before
# being passed to the regression model (using log(dim) instead of dim directly). This means the
# model is trained and will be making predictions in the linearized space.
#
# Dimension with shape
# ~~~~~~~~~~~~~~~~~~~~
#
# If some dimensions have a :ref:`search-space-shape` larger than 1, they will be flattened so that
# each subdimension can be represented in the bar plot.

# Load the data for the specified experiment
experiment = get_experiment("2-dim-shape-exp", storage=storage)
experiment.plot.lpi()

#%%
# In the example above, the dimension ``learning_rate~loguniform(1e-5, 1e-2, shape=3)``
# is flattened and represented with ``learning_rate[i]``. If the shape would be or more dimensions
# (ex: ``(3, 2)``), the indices would be ``learning_rate[i,j]`` with i=0..2 and j=0..1.

#%%
# Categorical dimension
# ~~~~~~~~~~~~~~~~~~~~~
#
# Categorical dimensions are converted into integer values, so that the regression model
# can handle them. The integers are simply indices that are assigned to each category in arbitrary
# order. Here is an example where dimension ``mt-join`` has the prior
# ``choices(['mean', 'max', 'concat'])``.

# Load the data for the specified experiment
experiment = get_experiment("3-dim-cat-shape-exp", storage=storage)
experiment.plot.lpi()

#%%
# Finally we save the image to serve as a thumbnail for this example. See
# the guide
# :ref:`How to save <sphx_glr_auto_examples_how-tos_code_2_how_to_save.py>`
# for more information on image saving.

fig.write_image("../../docs/src/_static/lpi_thumbnail.png")

# sphinx_gallery_thumbnail_path = '_static/lpi_thumbnail.png'

#%%
# .. [Biedenkapp2018] Biedenkapp, André, et al.
#    "Cave: Configuration assessment, visualization and evaluation."
#    International Conference on Learning and Intelligent Optimization.
#    Springer, Cham, 2018.
