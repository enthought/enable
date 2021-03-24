# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import functools
import time


def instrument_graphics_context(klass):
    """ Add profiling wrappers to all public methods of a class.

    This is intended to be used on implementers of `AbstractGraphicsContext`,
    but it could just as easily wrap other classes (although the "draw time"
    text in the profile log will look odd)

    To use, call this function on a `GraphicsContext` class object once at the
    module level. Then in a method which wraps a whole draw cycle (like
    `paintEvent` in Qt), add a call to the `dump_profile_and_reset` method
    after the drawing is done. Each paint will append profiling results to the
    output file.
    """
    # Wrap all the public methods
    for name in dir(klass):
        if name.startswith('_'):
            continue

        func = getattr(klass, name)
        if callable(func):
            setattr(klass, name, _get_wrapped(func))

    # Add the dumper method and the counter storage
    klass.dump_profile_and_reset = _dump_profile_and_reset
    klass._perf_counters = {}


def _dump_profile_and_reset(self, filename):
    """ Dump collected profiling results from an instrumented GC
    """
    data = [(k, v[0], v[1]) for k, v in self._perf_counters.items()]
    self._perf_counters = {}
    data.sort(key=lambda x: x[2], reverse=True)
    total = functools.reduce(lambda x, y: x + y[2], data, 0.0) * 1000
    fns = '\n'.join([f'{d[0]} {d[1]} {d[2]*1000:0.5f}' for d in data])

    text = f'\n*********\nTotal draw time: {total:0.5f}\n{fns}\n'
    # NOTE: We open the file in append mode
    with open(filename, "a+") as fp:
        fp.write(text)


def _get_wrapped(func):
    """ Wrap a single GC method
    """
    def wrapped_method(self, *args, **kwargs):
        key = func.__name__
        calls, duration = self._perf_counters.setdefault(key, (0, 0.0))
        start = time.perf_counter()
        retval = func(self, *args, **kwargs)
        delta = time.perf_counter() - start
        self._perf_counters[key] = (calls+1, duration+delta)
        return retval
    return wrapped_method
