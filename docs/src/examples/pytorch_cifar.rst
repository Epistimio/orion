**************************
Example with pytorch-cifar
**************************

.. note ::

    If database not setup or Oríon not installed, follow setup instructions: :doc:`../installing` and :doc:`../database`

pip3 install torch torchvision

git clone git@github.com:kuangliu/pytorch-cifar.git

cd pytorch-cifar

.. code-block:: bash

    sed -i '1 i\#!/usr/bin/env python' main.py

Add to last line of test()

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

Last line of the main() function

.. code-block:: python

        test_error_rate = test(epoch)

    report_results([dict(
        name='test_error_rate',
        type='objective',
        value=test_error_rate)])

.. code-block:: bash

    orion -v hunt -n resnet18-cifar10 python main.py --lr~'loguniform(1e-5, 1.0)'

.. note ::

    If you are using python3, the script will fail with message

    .. code-block:: bash

        ImportError: No module named 'vgg'

    You can fix this with the following command from the repository's root.

    .. code-block:: bash

        sed -i 's/from /from ./g' models/__init__.py


.. # orion submit -n resnet18-cifar10 mysubmissionfile
