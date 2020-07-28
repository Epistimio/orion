The REST API provides another way to retrieve information about the experiments. The REST server
serves requests to obtain information about trials and experiments.

The API serves the following endpoints.

* experiments/
* experiments/:name
* trials/:exp_name
* trials/:exp_name/:id
* plots/:kind


The REST server is started by the command line ``$ orion serve``. The service is hosted through
`gunicorn <https://gunicorn.org/>`. The database and host details are configured through the option
``--config <file>``. Storage options are expressed in the configuration file in the same fashion as
for other commands. Gunicorn options are placed under the top-level node ``gunicorn``.
