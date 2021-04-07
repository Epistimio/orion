*************
Visualization
*************

.. contents::
   :depth: 2
   :local:

Orion uses plotly to generate reports for experiments.
This section refers to the use of the `orion plot ...` subcommand.

For details about the use of the web api giving
similar plots, refer to the documentation :doc:`/user/web_api`.

=====
Usage
=====

The arguments expected by the "plot" subcommand are as follows

.. code-block :: console

   usage: orion plot [-h] [-n stringID] [-u USER] [-v VERSION] [-c path-to-config]
                     [-t {png,jpg,jpeg,webp,svg,pdf,html,json}]
                     [-o OUTPUT] [--scale SCALE]
                     {lpi,parallel_coordinates,partial_dependencies,regret}

   Produce plots for Oríon experiments

   positional arguments:
   {lpi,parallel_coordinates,partial_dependencies,regret}
                           kind of plot to generate.

   optional arguments:
   -h, --help            show this help message and exit
   -t {png,jpg,jpeg,webp,svg,pdf,html,json}, --type {png,jpg,jpeg,webp,svg,pdf,html,json}
                           type of plot to return. (default: png)
   -o OUTPUT, --output OUTPUT
                           path where plot is saved. Will override the `type` argument.
                           (default is {exp.name}-v{exp.version}_{kind}.{type})
   --scale SCALE         more pixels, but same proportions of the plot.
                           Scale acts as multiplier on height and width of resulting image.
                           Overrides value of 'scale' in plotly.io.write_image.

   Oríon arguments:
   These arguments determine orion's behaviour

   -n stringID, --name stringID
                           experiment's unique name;
                           (default: None - specified either here or in a config)
   -u USER, --user USER  user associated to experiment's unique name;
                           (default: $USER - can be overriden either here or in a config)
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

The plotting command requires a ``kind`` argument to determine which
of the four kinds of plots to generate.
The choice is between 'lpi', 'partial_dependencies', 'parallel_coordinates'
or 'regret'.


----
type
----

The ``type`` is basically the filename extension. This governs more than just the name of the file
because it determines the actual format of the output. The default is to give the user a 'png' file.

Behind the scenes, *plotly* generates an initial 'json' file, and renders it as an image
to be saved in the desired format. With ``type`` being 'json', that original file
is saved without rendering it to an image.

The accepted values of ``type`` are 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html' and 'json'.

------
output
------

The default value for the output filename is an automatically-generated
string formatted as "{experiment.name}_{kind}.{type}".
This also implies that the plot will be saved in the current directory.

When ``output`` is specified by the user with a file extension, the ``type`` argument
will be ignored because the value of ``type`` will be instead inferred by
the file extension in the ``output``.
For example, with ``--output=../myplot.jpg``, the results will be saved
in the parent directory and the ``type`` will be 'jpg'.

-----
scale
-----

With certain types of plots, it can be desirable to increase the
resolution of the output image in terms of pixel counts (equivalent to dpi).
This applies particularly to 'jpg' and 'png', but it does not affect 'json' or 'html'.

The reference value of ``scale`` is 1.0.
With ``--scale 2.0``, the height and width are going to be doubled.


