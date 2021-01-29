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

Learn how to get start using benchmark in Orion with this `sample notebook`_.

.. _sample notebook: https://github.com/Epistimio/orion/tree/develop/examples/benchmark/benchmark_get_start.ipynb
