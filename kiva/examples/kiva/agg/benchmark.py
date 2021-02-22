# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Benchmarks Agg rendering times.
"""
from time import perf_counter

from numpy import array, shape, arange, transpose, sin, cos, zeros, pi
from scipy import stats

import kiva
from kiva import agg


def benchmark_real_time(cycles=10, n_pts=1000, sz=(1000, 1000)):
    """ Render a sin wave to the screen repeatedly.  Clears
        the screen between each rendering.
    """
    print("realtime:", end=" ")
    width, height = sz
    pts = zeros((n_pts, 2), float)
    x = pts[:, 0]
    y = pts[:, 1]
    interval = width / float(n_pts)
    x[:] = arange(0, width, interval)
    t1 = perf_counter()
    # TODO: module 'kiva.agg' has no attribute 'GraphicsContextBitmap'
    gc = agg.GraphicsContextBitmap(sz)
    for i in range(cycles):
        y[:] = height / 2.0 + height / 2.0 * sin(
            x * 2 * pi / width + i * interval
        )
        # gc.clear()
        gc.lines(pts)
        gc.stroke_path()
        # agg.write_bmp_rgb24("sin%d.bmp" % i,gc.bitmap)
    t2 = perf_counter()
    tot_time = t2 - t1
    print("tot,per cycle:", tot_time, tot_time / cycles)


def benchmark_compiled_path(cycles=10, n_pts=1000, sz=(1000, 1000)):
    """ Render a sin wave to a compiled_path then display it repeatedly.
    """
    width, height = sz
    pts = zeros((n_pts, 2), float)
    x = pts[:, 0]
    y = pts[:, 1]
    interval = width / float(n_pts)
    x[:] = arange(0, width, interval)
    y[:] = height / 2.0 + height / 2.0 * sin(x * 2 * pi / n_pts)
    path = agg.CompiledPath()
    path.lines(pts)
    # path.move_to(pts[0,0],pts[0,1])
    # for x,y in pts[1:]:
    #    path.line_to(x,y)

    t1 = perf_counter()
    # TODO: module 'kiva.agg' has no attribute 'GraphicsContextBitmap'
    gc = agg.GraphicsContextBitmap(sz)
    for _ in range(cycles):
        # gc.clear()
        gc.add_path(path)
        gc.stroke_path()
    t2 = perf_counter()

    tot_time = t2 - t1
    print("tot,per cycle:", tot_time, tot_time / cycles)
    return


def benchmark_draw_path_flags(cycles=10, n_pts=1000, sz=(1000, 1000)):
    print("realtime:", end=" ")
    width, height = sz
    pts = zeros((n_pts, 2), float)
    x = pts[:, 0]
    y = pts[:, 1]
    interval = width / float(n_pts)
    x[:] = arange(0, width, interval)

    flags = [
        kiva.FILL,
        kiva.EOF_FILL,
        kiva.STROKE,
        kiva.FILL_STROKE,
        kiva.EOF_FILL_STROKE,
    ]

    for flag in flags:
        t1 = perf_counter()
        for i in range(cycles):
            # TODO: module 'kiva.agg' has no attribute 'GraphicsContextBitmap'
            gc = agg.GraphicsContextBitmap(sz)
            y[:] = height / 2.0 + height / 2.0 * sin(
                x * 2 * pi / width + i * interval
            )
            gc.lines(pts)
            gc.draw_path(flag)

        t2 = perf_counter()
        agg.write_bmp_rgb24("draw_path%d.bmp" % flag, gc.bitmap)
        tot_time = t2 - t1
        print("tot,per cycle:", tot_time, tot_time / cycles)
    return


def star_array(size=40):
    half_size = size * 0.5
    tenth_size = size * 0.1
    star_pts = [
        array((tenth_size, 0)),
        array((half_size, size - tenth_size)),
        array((size - tenth_size, 0)),
        array((0, half_size)),
        array((size, half_size)),
        array((tenth_size, 0)),
    ]
    return array(star_pts)


def circle_array(size=5):
    x = arange(0, 6.3, 0.1)
    pts = transpose(array((cos(x), sin(x)))).copy() * size / 2.0
    return pts


def star_path_gen(size=40):
    star_path = agg.CompiledPath()
    # spts = circle_array()
    spts = star_array(size)
    # star_path.lines(spts)
    star_path.move_to(spts[0][0], spts[0][1])
    for x, y in spts:
        star_path.line_to(x, y)
    star_path.close_path()
    return star_path


def benchmark_individual_symbols(n_pts=1000, sz=(1000, 1000)):
    "Draws some stars"
    # width, height = sz
    pts = stats.norm.rvs(size=(n_pts, 2)) * array(sz) / 8.0 + array(sz) / 2.0
    print(pts[5, :])
    print(shape(pts))
    star_path = star_path_gen()

    gc = agg.GraphicsContextArray(sz)
    gc.set_fill_color((1.0, 0.0, 0.0, 0.1))
    gc.set_stroke_color((0.0, 1.0, 0.0, 0.6))
    t1 = perf_counter()
    for x, y in pts:
        with gc:
            gc.translate_ctm(x, y)
            gc.add_path(star_path)
            gc.draw_path()
    t2 = perf_counter()
    gc.save("benchmark_symbols1.bmp")
    tot_time = t2 - t1
    print("star count, tot,per shape:", n_pts, tot_time, tot_time / n_pts)
    return


def benchmark_rect(n_pts=1000, sz=(1000, 1000)):
    "Draws a number of randomly-placed renctangles."
    # width, height = sz
    pts = stats.norm.rvs(size=(n_pts, 2)) * array(sz) / 8.0 + array(sz) / 2.0
    print(pts[5, :])
    print(shape(pts))

    gc = agg.GraphicsContextArray(sz)
    gc.set_fill_color((1.0, 0.0, 0.0, 0.1))
    gc.set_stroke_color((0.0, 1.0, 0.0, 0.6))
    t1 = perf_counter()
    for x, y in pts:
        with gc:
            gc.translate_ctm(x, y)
            gc.rect(-2.5, -2.5, 5, 5)
            gc.draw_path()
    t2 = perf_counter()
    gc.save("benchmark_rect.bmp")
    tot_time = t2 - t1
    print("rect count, tot,per shape:", n_pts, tot_time, tot_time / n_pts)
    return


def benchmark_symbols_all_at_once(n_pts=1000, sz=(1000, 1000)):
    """
    Renders all the symbols.
    """
    # width, height = sz
    pts = stats.norm.rvs(size=(n_pts, 2)) * array(sz) / 8.0 + array(sz) / 2.0
    star_path = agg.CompiledPath()
    star_path.lines(circle_array())

    gc = agg.GraphicsContextArray(sz)
    gc.set_fill_color((1.0, 0.0, 0.0, 0.1))
    gc.set_stroke_color((0.0, 1.0, 0.0, 0.6))
    path = agg.CompiledPath()
    t1 = perf_counter()
    for x, y in pts:
        path.save_ctm()
        path.translate_ctm(x, y)
        path.add_path(star_path)
        path.restore_ctm()
    gc.add_path(path)
    t2 = perf_counter()
    gc.draw_path()
    t3 = perf_counter()
    gc.save("benchmark_symbols2.bmp")
    build_path_time = t2 - t1
    render_path_time = t3 - t2
    tot_time = t3 - t1
    print(
        "star count, tot,building path, rendering path:",
        n_pts,
        tot_time,
        build_path_time,
        render_path_time,
    )
    return


def run_all_benchmarks(n_pts=1000, sz=(500, 500)):
    # TODO: does not work: Fix or remove?
    # benchmark_real_time(n_pts=n_pts, sz=sz)
    # TODO: does not work: Fix or remove?
    # benchmark_compiled_path(n_pts=n_pts, sz=sz)
    benchmark_individual_symbols(n_pts=n_pts, sz=sz)
    benchmark_rect(n_pts=n_pts, sz=sz)
    benchmark_symbols_all_at_once(n_pts=n_pts, sz=sz)
    # TODO: does not work: Fix or remove?
    # benchmark_draw_path_flags(n_pts=n_pts, sz=sz)


if __name__ == "__main__":
    run_all_benchmarks(n_pts=100, sz=(500, 500))
