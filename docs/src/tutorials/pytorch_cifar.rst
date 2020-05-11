***************
PyTorch CIFAR10
***************

.. note ::

    If Oríon not installed: pip install orion

    If the database is not setup, you can follow the instructions here:
    :doc:`/install/database`.

    Alternatively, you can test the example without setting up a database by
    using the option `--debug`, but note that all data gathered during an
    execution will be lost at the end of it.

Set up

.. code-block:: bash

    pip3 install torch torchvision

    git clone https://github.com/kuangliu/pytorch-cifar.git
    cd pytorch-cifar

Add to last line of test()

.. code-block:: python

    return 1 - (correct / len(test_loader.dataset))

Last line of the main() function

.. code-block:: python

        test_error_rate = test(epoch)

    report_objective(objective, name='test_error_rate')

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
