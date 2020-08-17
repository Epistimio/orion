The REST API provides another way to retrieve information about the experiments. The REST server
serves requests to obtain information about trials and experiments.

The API serves the following endpoints.

* experiments/
* experiments/:name
* trials/:exp_name
* trials/:exp_name/:id
* plots/:kind


The REST server is started by the command line ``$ orion serve``. The service is hosted through
`gunicorn <https://gunicorn.org/>`_. The database and host details are configured through the option
``--config <file>``. Storage options are expressed in the configuration file in the same fashion as
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

Attributes
~~~~~~~~~~

:title:
    The type of error. Can be one of ``Experiment not found``, ``Invalid parameter``,
    and ``Trial not found``.
:description:
    The human-readable description of the error.
