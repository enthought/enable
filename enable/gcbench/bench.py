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

_MAX_DURATION = 1.0
_SIZE = (512, 512)
_BACKENDS = {
    "ui": {
        "kiva.agg": "enable.null.image",
        "cairo": "enable.null.cairo",
        "celiagg": "enable.null.celiagg",
        "opengl": "enable.gcbench.opengl",
        "qpainter": "enable.null.qpainter",
        "quartz": "enable.null.quartz",
    },
    "file": {
        "pdf": "enable.null.pdf",
        "ps": "enable.null.ps",
        "svg": "enable.null.svg",
    },
}


def benchmark(outdir=None):
    """ Benchmark all backends
    """
    suite = gen_suite()

    results = {}
    # NOTE: Only checking UI backends for now
    for name, mod_name in _BACKENDS["ui"].items():
        print(f"Benchmarking backend: {name}", end="")
        try:
            module = importlib.import_module(mod_name)
        except ImportError:
            print(" ... Not available")
            continue
        results[name] = benchmark_backend(suite, name, module, outdir=outdir)

    return results


def benchmark_backend(suite, mod_name, module, outdir=None):
    """ Benchmark a single backend
    """
    GraphicsContext = getattr(module, "GraphicsContext")
    gc = GraphicsContext(_SIZE)

    timings = {}
    for name, symbol in suite.items():
        print(f"\n\tBenchmark {name}", end="")
        try:
            instance = symbol(gc, module)
        except Exception:
            continue

        if name.endswith("2x"):
            # Double sized
            with gc:
                gc.scale_ctm(2, 2)
                timings[name] = gen_timings(gc, instance)
        else:
            # Normal scale
            timings[name] = gen_timings(gc, instance)

        if timings[name] is None:
            print(f" ... Failed", end="")

        if timings[name] is not None and outdir is not None:
            fname = os.path.join(outdir, f"{mod_name}.{name}.png")
            gc.save(fname)

    print()  # End the line that was left
    return timings


def gen_suite():
    """ Create a suite of benchmarks to run against each backend
    """
    from enable.gcbench import suite

    benchmarks = {}
    for name in dir(suite):
        symbol = getattr(suite, name)
        if inspect.isclass(symbol):
            benchmarks[name] = symbol
            benchmarks[f"{name} 2x"] = symbol

    return benchmarks


def gen_timings(gc, func):
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
    return {
        "mean": times.mean() * 1000,
        "min": times.min() * 1000,
        "max": times.max() * 1000,
        "std": times.std() * 1000,
        "count": len(times),
    }
