Asynchronous Successive Halving Algorithm
=========================================

Can't build documentation because of import order.
Sphinx is loading ``orion.algo.asha`` before ``orion.algo`` and therefore
there is a cycle between the definition of ``OptimizationAlgorithm`` and
``ASHA`` as the meta-class ``Factory`` is trying to import ``ASHA``.
`PR #135 <https://github.com/Epistimio/orion/pull/135/files>`_ should get rid of this problem.
