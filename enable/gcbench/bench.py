# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import importlib
import inspect
import os
import time

import numpy as np

from enable.gcbench.data import BenchResult, BenchTiming

_MAX_DURATION = 1.0
_SIZE = (512, 512)
_2X_SIZE = (1024, 1024)
_BACKENDS = {
    "gui": {
        "kiva.agg": "enable.null.image",
        "cairo": "enable.null.cairo",
        "celiagg": "enable.null.celiagg",
        "opengl": "enable.gcbench.opengl",
        "qpainter": "enable.null.qpainter",
        "quartz": "enable.null.quartz",
    },
    "file": {
        "pdf": "enable.gcbench.pdf",
        "ps": "enable.null.ps",
        "svg": "enable.null.svg",
    },
}


def benchmark(outdir=None):
    """ Benchmark all backends
    """
    suite = gen_suite()
    results = {btype: {} for btype in _BACKENDS}

    for btype, backends in _BACKENDS.items():
        for name, mod_name in backends.items():
            print(f"Benchmarking backend: {name}", end="")
            try:
                module = importlib.import_module(mod_name)
            except ImportError:
                print(" ... Not available")
                continue

            if btype == "gui":
                # GUI backends are checked for performance (and features).
                results[btype][name] = benchmark_backend(
                    suite, name, module, outdir=outdir
                )
            else:
                # File backends are checked for feature coverage.
                # XXX: Use the fact that `name` is the same as the file ext.
                results[btype][name] = exercise_backend(
                    suite, name, module, extension=name, outdir=outdir
                )

    return results


def benchmark_backend(suite, mod_name, module, outdir=None):
    """ Benchmark a single backend
    """
    GraphicsContext = getattr(module, "GraphicsContext")

    results = {}
    for name, symbol in suite.items():
        # Result `summary` defaults to "fail"
        results[name] = result = BenchResult(output_size=_SIZE)

        size = _2X_SIZE if name.endswith("2x") else _SIZE
        gc = GraphicsContext(size)

        print(f"\n\tBenchmark {name}", end="")
        try:
            instance = symbol(gc, module)
        except Exception:
            print(f" ... Failed", end="")
            continue

        if name.endswith("2x"):
            # Double sized
            with gc:
                gc.scale_ctm(2, 2)
                timing = gen_timing(gc, instance)
        else:
            # Normal scale
            timing = gen_timing(gc, instance)

        if timing is None:
            print(f" ... Failed", end="")
            continue

        result.timing = timing
        result.summary = "success"
        if outdir is not None:
            fname = os.path.join(outdir, f"{mod_name}.{name}.png")
            gc.save(fname)
            result.output = os.path.basename(fname)

    print()  # End the line that was left
    return results


def exercise_backend(suite, mod_name, module, extension, outdir=None):
    """ Exercise a single backend
    """
    GraphicsContext = getattr(module, "GraphicsContext")

    results = {}
    for name, symbol in suite.items():
        # Result `summary` defaults to "fail"
        results[name] = result = BenchResult(output_size=_SIZE)

        # Skip 2x versions
        if name.endswith("2x"):
            result.summary = "skip"
            continue

        # Use a fresh context each time
        gc = GraphicsContext(_SIZE)

        print(f"\n\tBenchmark {name}", end="")
        try:
            instance = symbol(gc, module)
        except Exception:
            print(f" ... Failed", end="")
            continue

        try:
            instance()
            result.summary = "success"
        except Exception:
            print(f" ... Failed", end="")
            continue

        if outdir is not None:
            fname = os.path.join(outdir, f"{mod_name}.{name}.{extension}")
            gc.save(fname)
            # Record the output
            result.output = os.path.basename(fname)

    print()  # End the line that was left
    return results


def gen_suite():
    """ Create a suite of benchmarks to run against each backend
    """
    # Import here so we can use `suite` as a name elsewhere.
    from enable.gcbench import suite

    benchmarks = {}
    for name in dir(suite):
        symbol = getattr(suite, name)
        if inspect.isclass(symbol):
            benchmarks[name] = symbol
            benchmarks[f"{name}_2x"] = symbol

    return benchmarks


def gen_timing(gc, func):
    """ Run a function multiple times and generate some stats
    """
    duration = 0.0
    times = []
    while duration < _MAX_DURATION:
        gc.clear()
        t0 = time.perf_counter()
        try:
            func()
        except Exception:
            # Not all backends support everything
            break
        times.append(time.perf_counter() - t0)
        duration += times[-1]

    if not times:
        return None

    times = np.array(times)
    return BenchTiming(
        count=len(times),
        mean=times.mean() * 1000,
        minimum=times.min() * 1000,
        maximum=times.max() * 1000,
        stddev=times.std() * 1000,
    )
