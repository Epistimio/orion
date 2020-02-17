********************
Scikit-learn
********************
.. Might also think of moving this file to examples/ then we an example auto contained in the
   repository. We invoke this file from index.rst

In this example, we're going to demonstrate how Oríon can be integrated to a minimal model using
scikit-learn [scikit]_ on the iris dataset [iris]_. The files mentioned in this example are
available in the folder examples/scikit-learn-iris/.

The requirements are listed in requirements.txt. You can quickly install them using :command:`$ pip
install -r requirements.txt`. If you haven't installed Oríon previously, make sure to see
:doc:`configure it properly <../install/core>` before going further.

Original script
---------------

Our script looks like this

.. literalinclude:: ../../../examples/sklearn-iris/main.py
   :language: python
   :lines: 2-3, 5-23

This very basic script takes in parameter one positional argument for the hyper-parameter *epsilon*
which control the loss in the script

The script is divided in three parts:

#. Parsing of the script arguments
#. Loading and splitting the data, then training a classifier using the researcher-defined *epsilon*
#. Testing the classifier and reporting the accuracy

.. note::
   The workflow presented in the script is simplified compared to the reality on purpose. The
   objective of this example is to illustrate the basic steps to use Oríon.

To execute the experience, one would do ``$ ./main.py <epsilon>``, callling this script multiple
times to try different values for ``<epsilon>``.

.. caution::
   TODO

Adapting to Orion
-----------------

.. caution::
   TODO


.. [scikit] Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion,
            Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine Learning in Python"
            Journal of machine learning research 12, no. Oct (2011): 2825-2830.

.. [iris] Fisher, Ronald A. "The use of multiple measurements in taxonomic problems"
          Annals of eugenics 7, no. 2 (1936): 179-188.
