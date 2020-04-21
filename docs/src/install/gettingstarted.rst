***************
Getting Started
***************

Welcome! In this chapter, we give a quick overview of Oríon's main features and how it can help you
streamline your machine learning workflow whether you are a researcher or engineer.

Oríon is a black box function optimization library with a key focus on usability and integrability
for its users. For example, as an machine learning engineer, you can integrate Oríon to your
existing ML workflow to handle reliably the hyperparameter optimization and tuning phase. As a ML
researcher, you can use Oríon to tune your models but also integrate your own algorithms to Oríon to
serve as an efficient optimization engine and compare with other algorithms in the same context and
conditions.

Oríon is built to be non-intrusive, highly integrable, operate in parallel environments, and produce
reproducible experiments. It takes only one line of code to start using it in your existing
projects. It's natively asynchronous so it can perform efficiently on your laptop and in a computing
farm with thousands of processors, without the need to configure a master and workers. Oríon
supports the latest established hyperparameter algorithms out of the box, making it easy to switch
between them or create benchmarks. It also supports a vast range of search spaces. Oríon uses a
configuration agnostic approach where you can use any configuration file format you're comfortable
with. The results are always stored in a database that can be either in-memory, local or remote.
These results can be queried directly from the database itself or using convenient methods from
Oríon's API. Additionally, the experiments are versioned -- think of it as a git for scientific
experimentation -- enabling you to keep track of all your trials with their parameters. This
guarantees that you can reproduce or trace back the steps in your work for free. Finally, a plugin
system enables you to integrate and distribute your algorithms easily to other members of the
community.

Conversely, Oríon does not aim to be a machine learning framework or pipeline, or an automatic
machine learning product. Oríon focuses essentially on black box optimization. However, we do
encourage developers to integrate Oríon into that kind of systems as a component and we will do
our best to help you if you're interested.

Before continuing the overview, we assume that you have a basic understanding of machine learning
concepts and that you already have installed Oríon on your machine and selected a database. If it's
not the case please refer to our :doc:`installation instructions </install/core>` and :doc:`database
setup </install/database>`.

Integration
===========

Python API
----------

Optimize
========

Scaling up
----------

Search Space
============

The search space is defined by priors for each hyperparameter to optimize. In the snippet earlier,
we used the *loguniform* prior. We support almost all the distributions from `scipy
<https://docs.scipy.org/doc/scipy/reference/stats.html>`_ out of the box. You can define them either
directly in the command line (as shown previously) or in a configuration file:

.. code-block:: yaml

    lr: 'orion~loguniform(1e-5, 1.0)'

And then use it with:

.. code-block:: console

   $ orion hunt main.py --config config.yaml

Make sure to visit :doc:`/user/searchspace` for an exhaustive list of priors and their parameters.

Algorithms
==========

Similarly to search spaces, Oríon supports multiple algorithms out of the box: :ref:`random-search`,
:ref:`ASHA`, and :ref:`hyperband-algorithm`. Each one is fully configurable through the
configuration file. The samples of hyperparameter are based on the previous trials.

Make sure to checkout `this presentation
<https://docs.google.com/presentation/d/18g7Q4xRuhMtcVbwmFwDfH7v9gKS252-laOi9HrEQ7a4/present?slide=id.g6ba6d709b9_4_19>`_
for an quick overview of each algorithm and to visit :doc:´/user/algorithms´ to learn about the
algorithms and get recommendations about their use cases.


Monitoring
==========

Next steps
==========

It's worth to take a look at the :doc:`configuration system </user/config>` to learn more about how
to make the most out of Oríon and define precise behaviors for your algorithms and experiments.

Explore the :doc:`User Manual </user/overview>`, orion is simple from the outside but is feature
rich! If you're a researcher or develop you might be interested to :doc:`contribute
</developer/overview>` or develop your own :doc:`algorithms plugins </plugins/base>`!
