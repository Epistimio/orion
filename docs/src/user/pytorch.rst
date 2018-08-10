**************
Simple example
**************

Installation and setup
======================

First, install Oríon follwing :doc:`/install/core` and configure the database
(:doc:`/install/database`).  Then install `pytorch`, `torchvision` and clone the PyTorch
[examples repository](https://github.com/pytorch/examples):

.. code-block:: bash

    $ pip3 install torch torchvision
    $ git clone git@github.com:pytorch/examples.git


Adapting the code of MNIST example
==================================

.. code-block:: bash

    $ cd examples/mnist

In your favourite editor add a line `#!/usr/bin/env python` to the `main.py` and make it
executable, for example:

.. code-block:: bash

    $ sed -i '1s/^/#!/usr/bin/env python/' main.py
    $ chmod +x main.py

Add imports on top:

.. code-block:: python

    from orion.client import report_results

Propagate the performance of the model adding this to last line of test():

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

Report the performance to Oríon adding this to the last line of the main() function:

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
