..
   The REST API is documented using the sphinx extension sphinxcontrib.httpdomain
   https://sphinxcontrib-httpdomain.readthedocs.io/

The Oríon Web API is a `RESTful service <https://en.wikipedia.org/wiki/Representational_state_transfer>`_
that provides a way to retrieve and visualize information about your experiments and trials.

The API requests are handled by the Oríon itself.
The REST server is started by the command line ``$ orion serve``. The service is hosted through a
`gunicorn <https://gunicorn.org/>`_ container.
The database and host details are configured through the option ``--config <file>``.
Storage options are expressed in the configuration file in the same fashion as
for other commands. `Gunicorn options <https://docs.gunicorn.org/en/stable/settings.html>`_ are
specified under the top-level node ``gunicorn``. An example is included below:

.. code-block:: yaml

   storage:
      database:
         host: 'database.pkl'
         type: 'pickleddb'

   gunicorn:
      bind: '127.0.0.1:8000'
      workers: 4
      threads: 2

Authentication
---------------
No authentication is necessary at the moment to use the API.

Runtime
-------
The runtime resource represents the runtime information for the API server.

.. http:get:: /

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. sourcecode:: json

      {
        "orion": "v0.1.7",
        "server": "gunicorn",
        "database": "pickleddb"
      }

   :>json orion: The version of Oríon running the API server.
   :>json server: The WSGI HTTP Server hosting the API server.
   :>json database: The type of database where the HPO data is stored.


Experiments
-----------
The experiment resource permits the retrieval of in-progress and completed experiments. You can
retrieve individual experiments as well as a list of all your experiments.

.. http:get:: /experiments

   Return an unordered list of your experiments. Only the latest version of your experiments are
   returned.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      [
        {
         "name":"JCZY5",
         "version":2
        },
        {
         "name":"UGH3",
         "version":1
        }
      ]

   :>jsonarr name: Name of the experiment.
   :>jsonarr version: Latest version of the experiment.

.. http:get:: /experiments/:name

   Retrieve the details of the existing experiment named ``name``.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      {
         "name": "JCZY5",
         "version": 2,
         "status": "done",
         "trialsCompleted": 8,
         "startTime": "2020-01-21T16:29:33.73701",
         "endTime": "2020-01-22 14:43:42.02448",
         "user": "your username",
         "orionVersion": "0.1.7",
         "config": {
            "maxTrials": 10,
            "algorithm": {
               "name": "hyperband",
               "seed": 42,
               "repetitions": 1
            },
            "space": {
               "epsilon":"~uniform(1,5)",
               "lr":"~uniform(0.1,1)"
            }
         },
         "bestTrial": {
            "id": "f70277",
            "submitTime": "2020-01-22 14:19:42.02448",
            "startTime": "2020-01-22 14:20:42.02448",
            "endTime": "2020-01-22 14:20:42.0248",
            "parameters": {
               "epsilon": 1,
               "lr": 0.1
            },
           "objective": -0.7865584361152724,
           "statistics": {
               "low": 1,
               "high": 42
           }
         }
      }

   :query version: Optional version of the experiment to retrieve. If unspecified, the latest
      version of the experiment is retrieved.

   :>json name: The name of the experiment.
   :>json version: The version of the experiment.
   :>json status: The status of the experiment. Can be one of 'done' or 'not done' if there
      is trials remaining.
   :>json trialsCompleted: The number of trials completed.
   :>json startTime: The timestamp when the experiment started.
   :>json endTime: The timestamp when the experiment finished.
   :>json user: The name of the user that registered the experiment.
   :>json orionVersion: The version of Oríon that carried out the experiment.
   :>json config: The configuration of the experiment.
   :>json config.maxTrials: The trial budget for the experiment.
   :>json config.algorithm: The algorithm settings for the experiment.
   :>json config.space: The dictionary of priors as ``"prior-name":"prior-value"``.
   :>json bestTrial: The result of the optimization process in the form of the best trial.
      See the specification of :http:get:`/trials/:experiment/:id`.

   :statuscode 400: When an invalid query parameter is passed in the request.
   :statuscode 404: When the specified experiment doesn't exist in the database.

.. http:get:: /experiments/status/:name

   Retrieve the stats of the existing experiment named ``name``.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      {
        "trials_completed": 40,
        "best_trials_id": "955c77e7f567c2625f48546188a6cda1",
        "best_evaluation": -0.788720013597263,
        "start_time": "2019-11-25 16:02:02.872583",
        "finish_time": "2019-11-27 21:13:27.043519",
        "max_trials": 40,
        "nb_trials": 40,
        "progress": 1,
        "trial_status_count": {
          "completed": 40
        },
        "elapsed_time": "2 days, 5:11:24.006755",
        "sum_of_trials_time": "8 days, 23:15:15.594405",
        "eta": "0:00:00",
        "eta_milliseconds": 0
      }

   :query version: Optional version of the experiment to retrieve. If unspecified, the latest
      version of the experiment is retrieved.

   :>json trials_completed: The number of trials completed.
   :>json best_trial_id: The best trial ID.
   :>json best_evaluation: Best evaluation.
   :>json start_time: The timestamp when the experiment started.
   :>json finish_time: The timestamp when the experiment finished.
   :>json max_trials: The number of max trials for this experiment.
   :>json nb_trials: The current number of trials in this experiment.
   :>json progress: Floating value between 0 and 1 representing experiment progression.
   :>json trial_status_count: A dictionary mapping trial status to number of trials with this status in the experiment.
   :>json elapsed_time: The time elapsed since experiment started.
   :>json sum_of_trials_time: The sum of trials execution times.
   :>json eta: The estimation of remaining time for experiment to finish.
   :>json eta_milliseconds: The ETA in milliseconds (convenient for usages in Javascript).

   :statuscode 400: When an invalid query parameter is passed in the request.
   :statuscode 404: When the specified experiment doesn't exist in the database.

Trials
------

The trials resource permits the retrieval of your trials regardless of their status. You can
retrieve individual trials as well as a list of all your trials per experiment.

.. http:get:: /trials/:experiment

   Return an unordered list of the trials for the experiment ``experiment``.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      [
         {"id": "f70277"},
         {"id": "a5f7e1b"}
      ]

   :query ancestors: Optionally include the trials from all the experiment's parents.
      If unspecified, only the trials for this experiment version are retrieved.
   :query status: Optionally filter the trials by their status.
      See the available statuses in :py:func:`orion.core.worker.trial.validate_status`.
   :query version: Optional version of the experiment to retrieve. If unspecified, the latest
      version of the experiment is retrieved.

   :>jsonarr id: The ID of one trial for this experiment's version.

   :statuscode 400: When an invalid query parameter is passed in the request.
   :statuscode 404: When the specified experiment doesn't exist in the database.

.. http:get:: /trials/:experiment/:id

   Return the details of an existing trial with id ``id`` from the experiment ``experiment``.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      {
         "id": "f70277",
         "submitTime": "2020-01-22 14:19:42.02448"
         "startTime": "2020-01-22 14:20:42.02448",
         "endTime": "2020-01-22 14:20:42.0248",
         "parameters": {
            "epsilon": 1,
            "lr": 0.1
         },
         "objective": -0.7865584361152724,
         "statistics": {
            "low": 1,
            "high": 42
         }
      }

   :>json id: The ID of the trial.
   :>json submitTime: The timestamp when the trial was created
   :>json startTime: The timestamp when the trial started to be executed.
   :>json endTime: The timestamp when the trial finished its execution.
   :>json parameters: The dictionary of hyper-parameters as
      ``"parameter-name":"parameter-value"`` for this trial.
   :>json objective: The objective found for this trial with the given hyper-parameters.
   :>json statistics: The dictionary of statistics recorded during the trial
      as ``"statistic-name":"statistic-value"``.

   :statuscode 400: When an invalid query parameter is passed in the request.
   :statuscode 404: When the specified experiment doesn't exist in the database.
   :statuscode 404: When the specified trial doesn't exist for the specified experiment.

Plots
-----
The plot resource permits the generation and retrieval of `Plotly <https://plotly.com/>`_ plots to
visualize your experiments and their results.

.. http:get:: /plots/lpi/:experiment

   Return a lpi plot for the specified experiment.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   The JSON output is generated automatically according to the `Plotly.js schema reference <https://plotly.com/python/reference/index/>`_.

   :statuscode 404: When the specified experiment doesn't exist in the database.

.. http:get:: /plots/parallel_coordinates/:experiment

   Return a parallel coordinates plot for the specified experiment.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   The JSON output is generated automatically according to the `Plotly.js schema reference <https://plotly.com/python/reference/index/>`_.

   :statuscode 404: When the specified experiment doesn't exist in the database.

.. http:get:: /plots/partial_dependencies/:experiment

   Return a partial dependency plot for the specified experiment.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   The JSON output is generated automatically according to the `Plotly.js schema reference <https://plotly.com/python/reference/index/>`_.

   :statuscode 404: When the specified experiment doesn't exist in the database.

.. http:get:: /plots/regret/:experiment

   Return a regret plot for the specified experiment.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   The JSON output is generated automatically according to the `Plotly.js schema reference <https://plotly.com/python/reference/index/>`_.

   :statuscode 404: When the specified experiment doesn't exist in the database.


Benchmarks
----------
The benchmark resource permits the retrieval of in-progress and completed benchmarks. You can
retrieve individual benchmarks as well as a list of all your benchmarks.

.. http:get:: /benchmarks

   Return an unordered list of your benchmarks.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      [
        {
          "name": "branin_baselines",
          "algorithms": ["gridsearch", "random"],
          "assessments": {"AverageResult": {"repetitions": 2}},
          "tasks": {"Branin": {"max_trials": 10}},
        },
        {
          "name": "another_benchmark",
          "algorithms": ["gridsearch", {"random": {"seed": 1}}],
          "assessments": {
            "AverageRank": {"repetitions": 2},
            "AverageResult": {"repetitions": 2},
          },
          "tasks": {
            "Branin": {"max_trials": 10},
            "CarromTable": {"max_trials": 20},
            "EggHolder": {"dim": 4, "max_trials": 20},
          },
        },
      ]

   :>jsonarr name: Name of the benchmark.
   :>jsonarr algorithms: Algorithms of the benchmark.
   :>jsonarr assessments: Assessments used in the benchmark.
   :>jsonarr tasks: Tasks covered by the benchmark.

.. http:get:: /benchmarks/:name

   Retrieve the details of the existing benchmark named ``name``.

   **Example response**

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Content-Type: text/javascript

   .. code-block:: json

      {
        "name": "all_algos_webapi",
        "algorithms": ["gridsearch", {"random": {'seed': 1}}],
        "tasks": [{"Branin": {"max_trials": 5}}, {"RosenBrock": {"dim": 3, "max_trials": 5}}],
        "assessments": [{"AverageResult": {"repetitions": 3}}],
        "analysis": {
          "AverageResult": {
            "Branin": <plotly json encoding>
            "RosenBrock": <plotly json encoding>
          }
        }
      }

   :query asssessment: Optional, specific assessment to analyse. All assessments will appear as
                       part of the benchmark configuration, but the analysis dictionary will
                       only contain the specified assessment.
   :query task: Optional, specific task to analyse. All tasks will appear as
                part of the benchmark configuration, but the analysis dictionary will
                only contain the specified task.
   :query algorithms: Optional, specific algorithms to include in the analyse. Multiple
                values may be passed.
                All algorithms will appear as part of the benchmark configuration, but
                the analysis will be executed on the specified algorithms only.

   :>jsonarr name: Name of the benchmark.
   :>jsonarr algorithms: Algorithms of the benchmark.
   :>jsonarr assessments: Assessments used in the benchmark.
   :>jsonarr tasks: Tasks covered by the benchmark.
   :>jsonarr analysis: Dictionary of format {assessment_name: {task: <plotly json>}}

   :statuscode 400: When an invalid query parameter is passed in the request.
   :statuscode 404: When the specified benchmark does not exist in the database,
                    or assessment, task or algorithms are not part of the existing benchmark
                    configuration.

Errors
------
Oríon uses `conventional HTTP response codes <https://en.wikipedia.org/wiki/List_of_HTTP_status_codes>`_
to indicate the success or failure of an API request. In general, 2xx result codes indicate success
where 4xx indicate an error that failed given the information provided such as an unknown resource
or invalid parameters. 5xx codes indicate a server side error.

.. table:: HTTP Codes Summary

   ================ ====================================
   200 OK           The request succeeded
   400 Bad Request  Missing or invalid parameter
   404 Not Found    Resource unavailable or non-existent
   500 Server Error Internal server error
   ================ ====================================

:Response JSON Object:

   * **title** (string) - The type of error. Can be one of ``Experiment not found``,
     ``Invalid parameter``, and ``Trial not found``.
   * **description** (string) - The human-readable description of the error.
