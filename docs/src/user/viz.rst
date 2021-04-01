*************
Visualization
*************

.. contents::
   :depth: 1
   :local:

Orion uses plotly to generate reports for experiments.
This section refers to the use of the `orion plot ...` subcommand.

For details about the use of the web api giving
similar plots, refer to the documentation :doc:`/user/web_api`.

=====
Usage
=====

The arguments expected by the "plot" subcommand are as follows::

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

========================
More about the arguments
========================

The `kind` positional argument is the only mandatory argument.
Assuming that the user identified the experiment in the usual
way (e.g. using ``--name`` or a config file), the default behavior
is to generate the correct `kind` of plot, and to save it
as a "png" file in the current directory and with a filename
automatically formatted as "{experiment.name}_{kind}.png".

----
kind
----

The plotting command requires a ``kind`` argument to determine which of the four kinds of plots to generate.
The choice is between 'lpi', 'partial_dependencies', 'parallel_coordinates' or 'regret'.


----
type
----

The ``type`` is basically the filename extension. This governs more than just the name of the file
because it determines the actual format of the output. The default is to give the user a "png" file.

Behind the scenes, *plotly* generates an initial "json" file, and renders it as an image
to be saved in the desired format. With ``type`` being "json", that original file
is saved without rendering it to an image.

The accepted values of ``type`` are 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html' and 'json'.
