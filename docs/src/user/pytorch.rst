**************
Simple example
**************

Installation and setup
======================

Assume :doc:`/installing` and :doc:`/database` done.

.. code-block:: bash

    $ pip3 install torch torchvision
    $ git clone git@github.com:pytorch/examples.git


Adapting the code
=================

.. code-block:: bash
    
    $ cd examples/mnist

.. code-block:: bash

    $ sed -i '1s/^/#!/usr/bin/env python/' main.py
    $ chmod +x main.py

Add in top

.. code-block:: python

    from orion.client import report_results

Add to last line of test()

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

Last line of the main() function

.. code-block:: python

        test_error_rate = test(args, model, device, test_loader)
    
    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=test_error_rate)])

Execution
=========

.. code-block:: bash

    $ orion -v hunt -n lenet-mnist \
        ./main.py --lr~'loguniform(1e-5, 1.0)' --momentum~'uniform(0, 1)'


.. # orion submit -n resnet18-cifar10 mysubmissionfile


Analysis
========

TODO
