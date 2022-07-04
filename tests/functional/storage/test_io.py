"""
Test suite to measure I/O level during hyperparameter optimization.

If executed as a script, a graph is plotted using matplotlib and mean+-std I/O is logged in
terminal.

If executed with ``pytest``, ``test_io`` will verify if the level of I/O is close enough to nominal
levels.
"""
import argparse
import contextlib
import multiprocessing
import subprocess
import sys
import time
from collections import defaultdict, namedtuple

import bson
import numpy
from ptera import probing

from orion.client import build_experiment
from orion.client.runner import Runner
from orion.core.io.database.pickleddb import PickledDB
from orion.testing import OrionState


def foo(x, sleep_time):
    """Dummy function for the tests"""
    time.sleep(sleep_time)
    return [{"type": "objective", "name": "objective", "value": x}]


keys = ["net_in", "net_out"]
MongoStat = namedtuple("MongoStat", keys)

order_values = dict(b=1 / 1000.0, k=1, m=1000, g=1000**2)


def _convert_str_size(size):
    """Convert format 0b/0k/0m/0g to KB in float"""
    value = float(size[:-1])
    order = size[-1]
    return value * order_values[order]


class MongoStatMonitoring(multiprocessing.Process):
    def __init__(self, sleep_time=1):
        super().__init__()
        self.sleep_time = sleep_time
        self.q = multiprocessing.Queue()
        self.stop = multiprocessing.Event()

    def run(self):
        while not self.stop.is_set():
            row = mongostat()
            self.q.put(MongoStat(*(_convert_str_size(row[key]) for key in keys)))
            time.sleep(self.sleep_time)


def mongostat():
    """Return stat row of mongostat in a dict."""
    out = subprocess.run(
        "mongostat --rowcount=1".split(" "), stdout=subprocess.PIPE, check=True
    )
    header, row = out.stdout.decode("utf-8").split("\n")[:2]
    return dict(zip(header.split(), row.split()))


def max_trials(elements):
    """Max number of trials in a buffer"""
    if not elements:
        return 0
    return max(e["trials"] for e in elements)


@contextlib.contextmanager
def monitor_with_mongostat(interval=1, baseline_sleep=5):
    """Compute stats with MongoDB.

    This contextmanager is implemented to serve as a reference to validate the proper measure of I/O
    using ptera solely (`monitor_with_ptera`).
    """
    process = MongoStatMonitoring(sleep_time=interval)
    process.start()
    time.sleep(baseline_sleep)
    process.stop.set()
    process.join()

    baseline = defaultdict(int)
    while not process.q.empty():
        row = process.q.get(timeout=0.01)
        for i, key in enumerate(keys):
            baseline[key] += row[i]

    for key in keys:
        baseline[key] = numpy.array(baseline[key]).mean()

    data = ([], [], [])

    process = MongoStatMonitoring(sleep_time=interval)
    process.start()

    with probing("Runner.gather(trials) > #value") as prb:
        num_completed_trials = (
            prb.buffer_with_time(interval + 1).map(max_trials).accum()
        )

        yield data

    process.stop.set()
    process.join()

    baseline = defaultdict(int)
    while not process.q.empty():
        row = process.q.get(timeout=0.01)
        for i, key in enumerate(keys):
            data[i].append(row[i] - baseline[key])
        data[-1].append(num_completed_trials.pop(0))


def measure_size(element):
    """Measure size (KB) of an element encoded as BSON.

    List and tuples and converted elemented wize and the size is the sum of the elements sizes.
    """
    if not element:
        return 0

    if isinstance(element, (list, tuple)):
        return sum(measure_size(e) for e in element)

    # in KB
    return len(bson.BSON.encode(element)) / 1000.0


def measure_size_write(element):
    """Measure net_in/net_out of probing event on DB.write"""
    return {
        "net_in": sum(
            measure_size(element.get(key, None)) for key in ["data", "query"]
        ),
        "net_out": 0.001,
    }


def measure_size_read(element):
    """Measure net_in/net_out of probing event on DB.read"""
    return {
        "net_in": measure_size_count(element)["net_in"],
        "net_out": sum(measure_size(e) for e in element["#value"]),
    }


def measure_size_read_and_write(element):
    """Measure net_in/net_out of probing event on DB.read_and_write"""
    return {
        "net_in": measure_size_write(element)["net_in"],
        "net_out": measure_size(element["#value"]),
    }


def measure_size_count(element):
    """Measure net_in/net_out of probing event on DB.count"""
    return {
        "net_in": measure_size(element["query"]),
        "net_out": 0.001,
    }


def sum_stats(elements):
    """Sum statisticts in buffered probing events."""
    stats = defaultdict(float)
    for element in elements:
        for key in element:
            stats[key] += element[key]

    return stats


def get_online_desc():
    """Build compact inline stack of Orion call"""
    i = 1
    frame = sys._getframe(i)
    while (
        "orion/core/worker/experiment.py" not in frame.f_code.co_filename
        or frame.f_code.co_name.startswith("_")
    ):
        i += 1
        frame = sys._getframe(i)

    if frame.f_code.co_name == "acquire_algorithm_lock":
        caller_frame = sys._getframe(i + 2)
        caller = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}:{caller_frame.f_code.co_name}({frame.f_code.co_name})"
    else:
        caller = f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
    return caller


def get_full_stack_desc():
    """Build full stack of Orion call"""
    i = 1
    frame = sys._getframe(i)
    while (
        "orion/core/worker/experiment.py" not in frame.f_code.co_filename
        or frame.f_code.co_name.startswith("_")
    ):
        i += 1
        frame = sys._getframe(i)

    stack = []
    while not frame.f_code.co_filename.endswith(__file__):
        stack.append(
            f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
        )
        i += 1
        frame = sys._getframe(i)

    return "\n".join(stack)


def save_caller(element):
    """Save call stack in probing event"""

    element["stack"] = get_full_stack_desc()

    return element


@contextlib.contextmanager
def monitor_with_ptera(interval=1, db_backend_type=PickledDB, runner_type=Runner):
    """Monitor DB I/O and number of trials during optimization."""
    # NOTE: Need to have these classes imported for them to be resolved by ptera by name.
    db_backend = db_backend_type.__name__
    runner_name = runner_type.__name__
    with contextlib.ExitStack() as stack:
        selectors = dict(
            write=f"{db_backend}.write(query) > data",
            read=f"{db_backend}.read(query) > #value",
            read_and_write=f"{db_backend}.read_and_write(data, query) > #value",
            count=f"{db_backend}.count(query) > #value",
            n_trials=f"{runner_name}.gather(trials) > #value",
        )
        probes = dict()
        profiling = dict()
        for key, selector in selectors.items():
            probes[key] = stack.enter_context(probing(selector))

            if key == "n_trials":
                probes[key] = (
                    probes["n_trials"]
                    .buffer_with_time(interval)
                    .map(max_trials)
                    .accum()
                )
            else:
                measure_stream = probes[key].map(globals()[f"measure_size_{key}"])
                profiling[key] = measure_stream.map(save_caller).accum()
                probes[key] = (
                    measure_stream.buffer_with_time(interval).map(sum_stats).accum()
                )

        data = ([], [], [])
        yield data

    n_trials = probes.pop("n_trials")
    for interval_trials, row in zip(n_trials, zip(*probes.values())):
        if interval_trials == 0 and len(data[0]) > 0:
            interval_trials = data[2][-1]
        net_in = 0
        net_out = 0
        for probe in row:
            net_in += probe["net_in"]
            net_out += probe["net_out"]

        data[0].append(net_in)
        data[1].append(net_out)
        data[2].append(interval_trials)

    net_in_profiling = defaultdict(float)
    net_out_profiling = defaultdict(float)
    calls_profiling = defaultdict(int)
    for key, profiler in profiling.items():
        for element in profiler:
            net_in_profiling[element["stack"]] += element["net_in"]
            net_out_profiling[element["stack"]] += element["net_out"]
            calls_profiling[element["stack"]] += 1

    # TODO: Make this print optional
    return
    net_in_profiling = sorted(
        net_in_profiling.items(), key=lambda item: item[1], reverse=True
    )
    total = numpy.zeros(2)
    for item in net_in_profiling:
        print("net_in", calls_profiling[item[0]], item[1])
        print(item[0])
        total += numpy.array([calls_profiling[item[0]], item[1]])
    print("total", total)

    net_out_profiling = sorted(
        net_out_profiling.items(), key=lambda item: item[1], reverse=True
    )
    total = numpy.zeros(2)
    for item in net_out_profiling:
        print("net_out", calls_profiling[item[0]], item[1])
        print(item[0])
        total += numpy.array([calls_profiling[item[0]], item[1]])
    print("total", total)


monitoring_methods = dict(mongostat=monitor_with_mongostat, ptera=monitor_with_ptera)


colors = dict(joblib="#1f77b4", singleexecutor="#ff7f0e")


def main(argv=None):
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        default=["joblib", "singleexecutor"],
        nargs="+",
        type=str,
        help="Executor backends to use during the tests.",
    )
    parser.add_argument(
        "--n-base-trials",
        default=498,
        type=int,
        help=(
            "Number of trials to produce before starting the test. "
            "This measures the effect on I/O of last history in DB"
        ),
    )
    parser.add_argument(
        "--n-trials",
        default=2,
        type=int,
        help="Number of trials to execute during the test.",
    )
    parser.add_argument(
        "--trial-duration",
        default=30,
        type=float,
        help="Duration of trial execution (in seconds).",
    )
    parser.add_argument(
        "--output",
        default="test-io.png",
        type=str,
        help="File name to save figure.",
    )

    options = parser.parse_args(argv)

    for backend in options.backends:
        with OrionState():
            net_in, net_out, n_trials = compute_stats(
                monitoring_method="ptera",
                executor=backend,
                max_trials=(
                    options.n_base_trials,
                    options.n_base_trials + options.n_trials,
                ),
                sleep_time=options.trial_duration,
            )

        net_in = numpy.array(net_in)
        net_out = numpy.array(net_out)

        print(f"Backend: {backend}")
        print(
            f"Input: min={net_in.min()}, max={net_in.max()}, "
            f"mean={net_in.mean()}, std={net_in.std()}"
        )
        print(
            f"Output: min={net_out.min()}, max={net_out.max()}, "
            f"mean={net_out.mean()}, std={net_out.std()}"
        )

        max_points = min(map(len, [net_in, net_out, n_trials]))
        plt.plot(
            list(range(max_points)),
            net_in[:max_points],
            label=f"{backend}-in",
            color=colors[backend],
            linestyle="dashed",
        )
        plt.plot(
            list(range(max_points)),
            net_out[:max_points],
            label=f"{backend}-out",
            color=colors[backend],
        )

    plt.xlabel("Time (s)")
    plt.ylabel("I/O (KB/s)")

    plt.legend()
    plt.savefig(options.output)


def compute_stats(
    monitoring_method="ptera",
    executor="joblib",
    max_trials=(498, 500),
    sleep_time=30,
):
    experiment = build_experiment(
        f"test-io-{executor}-{monitoring_method}",
        space=dict(x="uniform(0, 1, precision=100)"),
        max_trials=max_trials[1],
    )

    with experiment.tmp_executor(executor, n_workers=1):
        experiment.workon(
            foo,
            max_trials=max_trials[1],
            max_trials_per_worker=max_trials[0],
            sleep_time=0.0001,
        )

        with monitoring_methods[monitoring_method]() as data:
            experiment.workon(
                foo,
                max_trials=max_trials[1],
                max_trials_per_worker=max_trials[1] - max_trials[0],
                sleep_time=sleep_time,
            )

    return data


def test_io():
    """Verify that I/O levels during optimization are close enough to nominal levels"""

    with OrionState():
        net_in, net_out, n_trials = compute_stats(
            monitoring_method="ptera",
            executor="joblib",
            max_trials=(198, 200),
            sleep_time=15.0,
        )
    net_in = numpy.array(net_in)
    net_out = numpy.array(net_out)

    NOMINAL_IN_MEAN = 16.45  # KB/s
    NOMINAL_IN_STD = 77.57  # KB/s

    NOMINAL_OUT_MEAN = 57.56  # KB/s
    NOMINAL_OUT_STD = 276.41  # KB/s

    assert net_in.mean() < NOMINAL_IN_MEAN + NOMINAL_IN_STD / numpy.sqrt(
        net_in.shape[0]
    )
    assert net_out.mean() < NOMINAL_OUT_MEAN + NOMINAL_OUT_STD / numpy.sqrt(
        net_out.shape[0]
    )


if __name__ == "__main__":
    main()
