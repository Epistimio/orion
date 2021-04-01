*************
Visualization
*************

.. contents::
   :depth: 2
   :local:

Orion uses plotly to generate reports for experiments.
This section refers to the use of the `orion plot ...` subcommand.

For details about the use of the web api giving
similar plots, refer to the documentation INSERT_LINK.

Usage
=====

.. code-block:: bash
    usage: orion plot [-h] [-n stringID] [-u USER] [-v VERSION] [-c path-to-config] [-t TYPE] [-o OUTPUT] [--scale SCALE] kind

    Produce plots for Oríon experiments

    positional arguments:
    kind                  kind of plot to generate. Pick one among ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']

    optional arguments:
    -h, --help            show this help message and exit
    -t TYPE, --type TYPE  type of plot to return. Pick one among ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'] (default: png)
    -o OUTPUT, --output OUTPUT
                            path where plot is saved. Will override the `type` argument. (default is {exp.name}_{kind}.{type})
    --scale SCALE         more pixels, but same proportions of the plot. Reference is 1.0. Overrides value of 'scale' in plotly.io.write_image.

    Oríon arguments:
    These arguments determine orion's behaviour

    -n stringID, --name stringID
                            experiment's unique name; (default: None - specified either here or in a config)
    -u USER, --user USER  user associated to experiment's unique name; (default: $USER - can be overriden either here or in a config)
    -v VERSION, --version VERSION
                            specific version of experiment to fetch; (default: None - latest experiment.)
    -c path-to-config, --config path-to-config
                            user provided orion configuration file

More about the arguments
========================

Bla bla bla