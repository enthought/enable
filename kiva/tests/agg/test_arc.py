# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from math import pi
import os
import tempfile
import unittest

import numpy

from kiva import agg


def draw_arcs(gc, x2, y2, radiusstep=25.0):
    gc.set_stroke_color((0.2, 0.2, 0.2))  # lightgray
    gc.move_to(0, 0)
    gc.line_to(100, 0)
    gc.line_to(x2, y2)
    gc.stroke_path()
    gc.set_stroke_color((0, 0, 0))
    for i in range(7):
        gc.move_to(0, 0)
        gc.arc_to(100, 0, x2, y2, i * radiusstep + 20.0)
    gc.stroke_path()


class TestAffineMatrix(unittest.TestCase):
    def test_arc_to(self):
        gc = agg.GraphicsContextArray((640, 480), "rgba32")
        axes = agg.CompiledPath()
        axes.move_to(0.5, 50.5)
        axes.line_to(100.5, 50.5)
        axes.move_to(50.5, 0.5)
        axes.line_to(50.5, 100.5)

        box = agg.CompiledPath()
        box.move_to(0.5, 0.5)
        box.line_to(100.5, 0.5)
        box.line_to(100.5, 100.5)
        box.line_to(0.5, 100.5)
        box.close_path()

        arc = agg.CompiledPath()
        arc.move_to(10, 10)
        arc.line_to(20, 10)
        arc.arc_to(40, 10, 40, 30, 20.0)
        arc.line_to(40, 40)

        whole_shebang = agg.CompiledPath()
        whole_shebang.save_ctm()
        whole_shebang.add_path(axes)
        whole_shebang.add_path(box)
        whole_shebang.translate_ctm(0.0, 50.5)
        whole_shebang.add_path(arc)

        whole_shebang.translate_ctm(50.5, 50.5)
        whole_shebang.rotate_ctm(-agg.pi / 2)
        whole_shebang.add_path(arc)
        whole_shebang.rotate_ctm(agg.pi / 2)

        whole_shebang.translate_ctm(50.5, -50.5)
        whole_shebang.rotate_ctm(-agg.pi)
        whole_shebang.add_path(arc)
        whole_shebang.rotate_ctm(agg.pi)

        whole_shebang.translate_ctm(-50.5, -50.5)
        whole_shebang.rotate_ctm(-3 * agg.pi / 2)
        whole_shebang.add_path(arc)
        whole_shebang.restore_ctm()

        gc.set_stroke_color((1.0, 0.0, 0.0))
        gc.set_line_width(1.0)

        ctm1 = gc.get_ctm()
        gc.translate_ctm(50.5, 300.5)
        gc.add_path(whole_shebang)
        gc.stroke_path()

        gc.translate_ctm(130.5, 50.0)
        ctm2 = gc.get_ctm()
        gc.rotate_ctm(-agg.pi / 6)
        gc.add_path(whole_shebang)
        gc.set_stroke_color((0.0, 0.0, 1.0))
        gc.stroke_path()
        gc.set_ctm(ctm2)

        gc.translate_ctm(130.5, 0.0)
        ctm2 = gc.get_ctm()
        gc.rotate_ctm(-agg.pi / 3)
        gc.scale_ctm(1.0, 2.0)
        gc.add_path(whole_shebang)
        gc.stroke_path()
        gc.set_ctm(ctm1)

        ctm1 = gc.get_ctm()
        gc.translate_ctm(150.5, 20.5)
        draw_arcs(gc, 70.5, 96.5)
        gc.translate_ctm(300.5, 0)
        draw_arcs(gc, 160.5, 76.5, 50.0)
        gc.set_ctm(ctm1)

        gc.translate_ctm(120.5, 100.5)
        gc.scale_ctm(-1.0, 1.0)
        draw_arcs(gc, 70.5, 96.5)
        gc.translate_ctm(-300.5, 100.5)
        gc.scale_ctm(0.75, -1.0)
        draw_arcs(gc, 160.5, 76.5, 50.0)
        with tempfile.TemporaryDirectory() as dirpath:
            gc.save(os.path.join(dirpath, "arc_to.png"))

    def test_arc(self):
        gc = agg.GraphicsContextArray((640, 648))
        with tempfile.TemporaryDirectory() as dirpath:
            gc.save(os.path.join(dirpath, "arc.png"))

    def test_skewing_matrix(self):
        val = agg.skewing_matrix(pi / 4.0, pi / 4.0)
        desired = numpy.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        actual = val.asarray()
        assert numpy.allclose(desired, actual)
