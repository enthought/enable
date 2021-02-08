# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Needed Tests

    Joins
        DONE 1. Each version same for all paths through aliased code.
        DONE 2. Each version same for all paths through antialiased code.

    Dashed Lines
        *. Should be tested for all versions of aliasing and all versions
           of join, cap.


    Clipping
        <Perhaps in different file>
        1. Test that clip_to_rect is inclusive on lower end and exclusive
           on upper end.
        2. Test that clip_to_rect behaves intelligently under ctm transforms.

    Note: There are numerous comments in code that refer to implementation
          details (outline, outline_aa, scanline_aa, etc.) from the C++
          code.  These are largely to help keep track of what paths have
          been tested.
"""
import unittest

from numpy import array

from kiva.agg import GraphicsContextArray
import kiva

from .test_utils import Utils


class JoinStrokePathTestCase(unittest.TestCase, Utils):
    def helper(self, antialias, width, line_cap, line_join, size=(10, 10)):

        gc = GraphicsContextArray(size, pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(1, 3)
        gc.line_to(7, 3)
        gc.line_to(7, 9)

        # Settings allow the faster outline path through C++ code
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(antialias)
        gc.set_line_width(width)
        gc.set_line_cap(line_cap)
        gc.set_line_join(line_join)

        gc.stroke_path()
        return gc

    def _test_alias_miter(self):
        """ fix me: This is producing aliased output.

                    I am not sure what this array should look like
                    exactly, so we just force it to fail right now
                    and print out the results.
        """
        antialias = False
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_MITER

        gc = self.helper(antialias, width, line_cap, line_join)

        actual = gc.bmp_array[:, :, 0]
        assert 0, "join=miter, width=3, antialias=False\n%s" % actual
        # assert_arrays_equal(actual, desired)

    def _test_alias_bevel(self):
        """ fix me: This is producing a line width of 4 instead of 3.

                    I am not sure what this array should look like
                    exactly, so we just force it to fail right now
                    and print out the results.
        """
        antialias = False
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_BEVEL

        gc = self.helper(antialias, width, line_cap, line_join)
        actual = gc.bmp_array[:, :, 0]
        assert 0, "join=bevel, width=3, antialias=False\n%s" % actual
        # assert_arrays_equal(actual, desired)

    def _test_alias_round(self):
        """ fix me: This is producing a line width of 4 instead of 3.

                    Also, the corner doesn't look so round. I have
                    checked that the scanline_aa renderer is setting
                    the stroked_path.line_join() value to agg::round_join.

                    I am not sure what this array should look like
                    exactly, so we just force it to fail right now
                    and print out the results.
        """
        antialias = False
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_ROUND

        gc = self.helper(antialias, width, line_cap, line_join)
        actual = gc.bmp_array[:, :, 0]
        assert 0, "join=round, width=3, antialias=False\n%s" % actual
        # assert_arrays_equal(actual, desired)

    def test_antialias_miter(self):
        """ fix me: How to make this pass on OS X and agg...
        """
        antialias = True
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_MITER

        gc = self.helper(antialias, width, line_cap, line_join)

        actual = gc.bmp_array[:, :, 0]
        desired = array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 127, 127, 127, 127,  63,   0,   0, 127, 255],
                [255,   0,   0,   0,   0,   0,   0,   0, 127, 255],
                [255,   0,   0,   0,   0,   0,   0,   0, 127, 255],
                [255, 127, 127, 127, 127, 127, 127, 127, 191, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            ]
        )
        self.assertRavelEqual(actual, desired)

    def test_antialias_bevel(self):
        antialias = True
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_BEVEL

        gc = self.helper(antialias, width, line_cap, line_join)
        actual = gc.bmp_array[:, :, 0]
        desired = array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 127,    0,  0, 127, 255],
                [255, 255, 255, 255, 255, 127,    0,  0, 127, 255],
                [255, 255, 255, 255, 255, 127,    0,  0, 127, 255],
                [255, 255, 255, 255, 255, 127,    0,  0, 127, 255],
                [255, 127, 127, 127, 127,  63,    0,  0, 127, 255],
                [255,   0,   0,   0,   0,   0,    0,  0, 127, 255],
                [255,   0,   0,   0,   0,   0,    0, 31, 223, 255],
                [255, 127, 127, 127, 127, 127, 127, 223, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            ]
        )
        self.assertRavelEqual(actual, desired)

    def test_antialias_round(self):
        """ fix me: How to make this test work for multiple renderers?
        """
        antialias = True
        width = 3
        line_cap = kiva.CAP_BUTT
        line_join = kiva.JOIN_ROUND

        gc = self.helper(antialias, width, line_cap, line_join)
        actual = gc.bmp_array[:, :, 0]
        desired = array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 255, 255, 255, 255, 127,   0,   0, 127, 255],
                [255, 127, 127, 127, 127,  63,   0,   0, 127, 255],
                [255,   0,   0,   0,   0,   0,   0,   0, 127, 255],
                [255,   0,   0,   0,   0,   0,   0,   0, 180, 255],
                [255, 127, 127, 127, 127, 127, 127, 179, 253, 255],
                [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            ]
        )
        self.assertRavelEqual(actual, desired)
