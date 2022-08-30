**********
Benchmark
**********

You can benchmark the performance of search algorithms with different tasks at different
assessment levels.

Benchmark can be created as below, refer to :doc:`/code/benchmark/benchmark_client`
for how to create and :doc:`/code/benchmark` for how to use benchmark.

.. code-block:: python

  from orion.benchmark.benchmark_client import get_or_create_benchmark
  from orion.benchmark.assessment import AverageResult, AverageRank
  from orion.benchmark.task import RosenBrock, EggHolder, CarromTable

  benchmark = get_or_create_benchmark(name='benchmark',
          algorithms=['random', 'tpe'],
          targets=[
              {
                  'assess': [AverageResult(2), AverageRank(2)],
                  'task': [RosenBrock(25, dim=3), EggHolder(20, dim=4), CarromTable(20)]
              }
          ])

Beside out of box :doc:`/code/benchmark/task` and :doc:`/code/benchmark/assessment`,
you can also extend benchmark to add new ``Tasks`` with :doc:`/code/benchmark/task/base` and
``Assessments`` with :doc:`/code/benchmark/assessment/base`,

To run benchmark with task :doc:`/code/benchmark/task/hpobench`, use ``pip install orion[hpobench]``
to install the extra requirements. HPOBench provides local and containerized benchmarks, but different
local benchmark could ask for total difference extra requirements. With ``pip install orion[hpobench]``,
you will be able to run local benchmark ``benchmarks.ml.tabular_benchmark``.
If you want to run other benchmarks in local, refer HPOBench `Run a Benchmark Locally`_. To run
containerized benchmarks, you will need to install `singularity`_.

Learn how to get start using benchmark in Orion with this `sample notebook`_.

.. _Run a Benchmark Locally: https://github.com/automl/HPOBench#run-a-benchmark-locally
.. _singularity: https://singularity.hpcng.org/admin-docs/master/installation.html
.. _sample notebook: https://github.com/Epistimio/orion/tree/develop/examples/benchmark/benchmark_get_start.ipynb
