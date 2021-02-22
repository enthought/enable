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

    Tasks
    -----
    DONE *. Change all rect classes to use a sequence instead of 4 separate
            args
    *. Implement a clear_rect that takes a rect argument and a color.
    *. Rename the clear() method to ?erase()? so that clear can stick with
       the clear_rect semantics.
    *. Implement arc.

    *. Implement a clear_rects that takes a rect list and a color.
    *. Implement clipping to multiple rectangles.

    Needed Tests
    ------------

    Coordinate system offset.
        DONE 1. Test that aliased line at zero puts full energy in last
                row of buffer.
        DONE 2. Test that antialiased line at zero puts half energy in
                last row of buffer.

    Aliased
        DONE 1. For simple line, test all paths through rendering pipeline.

    Antialiased
        DONE 1. For simple line, test all paths through rendering pipeline.

    Caps
        DONE 1. Each version same for all paths through aliased code.
        2. Each version same for all paths through antialiased code.

    Joins (in join_stroke_path_test_case.py)
        DONE 1. Each version same for all paths through aliased code.
        DONE 2. Each version same for all paths through antialiased code.

    Dashed Lines
        *. Should be tested for all versions of aliasing and all versions
           of join, cap.

    Curved Lines
        DONE *. curve_to

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


class StrokePathTestCase(unittest.TestCase, Utils):
    def test_alias_width_one(self):
        """ The fastest path through the stroke path code is for aliased
            path with width=1.  It is reasonably safe here not to worry
            with testing all the CAP/JOIN combinations because they are
            all rendered the same for this case.

            It is handled by the agg::rasterizer_outline and the
            agg::renderer_primitives classes in C++.

            Energy for an aliased horizontal line of width=1 falls
            within a single line of pixels.  With y=0 for this line, the
            engine should paint a single row of zeros along the
            bottom edge of the bmp array.
        """
        gc = GraphicsContextArray((3, 2), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(0, 0)
        gc.line_to(3, 0)

        # These settings allow the fastest path.
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(False)
        gc.set_line_width(1)

        gc.stroke_path()

        # test a single color channel.
        desired = array(((255, 255, 255), (0, 0, 0)))
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(actual, desired)

    def test_alias_width_two_outline_aa(self):
        """ When  width>1, alias text is drawn using a couple of different
            paths through the underlying C++ code. This test the faster of the
            two which uses the agg::rasterizer_outline_aa C++ code.  It is only
            used when 2<=width<=10, and cap is ROUND or BUTT, and join is MITER

            The C++ classes used in the underlying C++ code for this is
            agg::rasterizer_outline_aa and agg::renderer_outline_aa
        """
        gc = GraphicsContextArray((3, 2), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(0, 0)
        gc.line_to(3, 0)

        # Settings allow the 2nd fastest path.
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(False)
        gc.set_line_width(2)
        gc.set_line_cap(kiva.CAP_ROUND)
        gc.set_line_join(kiva.JOIN_MITER)

        gc.stroke_path()

        # test a single color channel.
        desired = array(((255, 255, 255), (0, 0, 0)))
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(actual, desired)

    def test_alias_width_two_scanline_aa(self):
        """ When width > 1, alias text is drawn using a couple of different
            paths through the underlying C++ code. This test the slower of the
            two which uses the agg::rasterizer_scanline_aa C++ code.  We've set
            the line join to bevel to trigger this path.
        """
        gc = GraphicsContextArray((3, 2), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(0, 0)
        gc.line_to(3, 0)

        # Settings allow the 2nd fastest path.
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(False)
        gc.set_line_width(2)
        gc.set_line_cap(kiva.CAP_ROUND)
        gc.set_line_join(kiva.JOIN_BEVEL)

        gc.stroke_path()

        # test a single color channel.
        desired = array(((255, 255, 255), (0, 0, 0)))
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(actual, desired)

    #########################################################################
    # Cap Tests
    #########################################################################

    def cap_equality_helper(self, antialias, width, line_cap, line_join,
                            size=(6, 6)):

        gc = GraphicsContextArray(size, pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(2, 3)
        gc.line_to(4, 3)

        # Settings allow the faster outline path through C++ code
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(antialias)
        gc.set_line_width(width)
        gc.set_line_cap(line_cap)
        gc.set_line_join(line_join)

        gc.stroke_path()
        return gc

    def test_alias_cap_round(self):
        """ Round caps should extend beyond the end of the line.  We
            don't really test the shape here.  To do this, a test of
            a wider line would be needed.

            fix me: This is rendering antialiased end points currently.
        """
        gc = GraphicsContextArray((6, 6), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(2, 3)
        gc.line_to(4, 3)

        # Set up line
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(False)
        gc.set_line_width(2)
        gc.set_line_cap(kiva.CAP_ROUND)
        gc.set_line_join(kiva.JOIN_MITER)

        gc.stroke_path()

        desired = array(
            (
                (255, 255, 255, 255, 255, 255),
                (255, 255, 255, 255, 255, 255),
                (255,   0,   0,   0,   0, 255),
                (255,   0,   0,   0,   0, 255),
                (255, 255, 255, 255, 255, 255),
                (255, 255, 255, 255, 255, 255),
            )
        )

        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_alias_cap_round_equality(self):
        """ There are two paths that can generate aliased round capped lines.
            This tests that they generate equivalent results

            fix me: This test fails because the two rendering paths
                    render the end caps differently.
        """
        antialias = False
        width = 2
        cap = kiva.CAP_ROUND
        # join=miter allows the faster outline path through C++ code.
        gc1 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_MITER)

        # join=bevel forces the scanline path through C++ code.
        gc2 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_BEVEL)

        # Instead of testing against a known desired value, we are simply
        # testing for equality...
        self.assertRavelEqual(gc1.bmp_array[:, :, 0], gc2.bmp_array[:, :, 0])

    def test_alias_cap_square(self):
        """ Square caps should extend beyond the end of the line. by
            half the width of the line.
        """
        gc = GraphicsContextArray((6, 6), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(2, 3)
        gc.line_to(4, 3)

        # Set up line
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(False)
        gc.set_line_width(2)
        gc.set_line_cap(kiva.CAP_SQUARE)
        gc.set_line_join(kiva.JOIN_MITER)

        gc.stroke_path()

        desired = array(
            (
                (255, 255, 255, 255, 255, 255),
                (255, 255, 255, 255, 255, 255),
                (255,   0,   0,   0,   0, 255),
                (255,   0,   0,   0,   0, 255),
                (255, 255, 255, 255, 255, 255),
                (255, 255, 255, 255, 255, 255),
            )
        )

        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_alias_cap_butt_equality(self):
        """ There are two paths that can generate aliased butt capped lines.
            This tests that they generate equivalent results
        """
        antialias = False
        width = 2
        cap = kiva.CAP_BUTT
        # join=miter allows the faster outline path through C++ code.
        gc1 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_MITER)

        # join=bevel forces the scanline path through C++ code.
        gc2 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_BEVEL)

        # Instead of testing against a known desired value, we are simply
        # testing for equality...
        self.assertRavelEqual(gc1.bmp_array, gc2.bmp_array)

    def test_alias_cap_square_equality(self):
        """ There are two paths that can generate aliased square capped lines.
            This tests that they generate equivalent results.
        """
        antialias = False
        width = 2
        cap = kiva.CAP_SQUARE
        # join=miter allows the faster outline path through C++ code.
        gc1 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_MITER)

        # join=bevel forces the scanline path through C++ code.
        gc2 = self.cap_equality_helper(antialias, width, cap, kiva.JOIN_BEVEL)

        # Instead of testing against a known desired value, we are simply
        # testing for equality...
        self.assertRavelEqual(gc1.bmp_array, gc2.bmp_array)

    def test_antialias_width_one(self):
        """ An anti-aliased horizontal line of width=1 has its energy
            centered between the bottom row of pixels and the next lower
            row of pixels (which is off the page).  It dumps half
            its energy in each, so we end up with a single line of
            127,127,127 pixel values in the last row of pixels.

            This particular set of flags is handled by the (fast)
            agg::rasterizer_outline_aa path through the C++ code.

        """
        gc = GraphicsContextArray((3, 2), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(0, 0)
        gc.line_to(3, 0)

        # Set up stroke
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(True)
        gc.set_line_width(1)
        gc.set_line_cap(kiva.CAP_BUTT)
        gc.set_line_join(kiva.JOIN_MITER)

        gc.stroke_path()

        # test a single color channel.
        desired = array(((255, 255, 255), (127, 127, 127)))
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_antialias_width_slower_path(self):
        """ An anti-aliased horizontal line of width=1 has its energy
            centered between the bottom row of pixels and the next lower
            row of pixels (which is off the page).  It dumps half
            its energy in each, so we end up with a single line of
            127,127,127 pixel values in the last row of pixels.

            This particular set of flags is handled by the
            agg::rasterizer_scanline_aa path through the C++ code.

        """
        gc = GraphicsContextArray((3, 2), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        gc.move_to(0, 0)
        gc.line_to(3, 0)

        # Set up stroke
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(True)
        gc.set_line_width(1)
        gc.set_line_cap(kiva.CAP_BUTT)
        gc.set_line_join(kiva.JOIN_BEVEL)

        gc.stroke_path()

        # test a single color channel.
        desired = array(((255, 255, 255), (127, 127, 127)))
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_curve_to(self):
        """ curve_to

            conv_curve happens early in the agg rendering pipeline,
            so it isn't neccessary to test every combination of
            antialias, line_cap, line_join, etc.  If it works for
            one, we should be in good shape for the others (until
            the implementation is changed of course...)

        """
        gc = GraphicsContextArray((10, 10), pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        # single horizontal line across bottom of buffer
        x0, y0 = 1.0, 5.0
        x1, y1 = 4.0, 9.0
        x2, y2 = 6.0, 1.0
        x3, y3 = 9.0, 5.0
        gc.move_to(x0, y0)
        gc.curve_to(x1, y1, x2, y2, x3, y3)

        # Set up stroke
        gc.set_stroke_color((0.0, 0.0, 0.0))  # black
        gc.set_antialias(True)
        gc.set_line_width(1)
        gc.set_line_cap(kiva.CAP_BUTT)
        gc.set_line_join(kiva.JOIN_MITER)

        gc.stroke_path()

        gc.set_stroke_color((0.0, 1.0, 1.0))
        gc.move_to(x0, y0)
        gc.line_to(x1, y1)
        gc.move_to(x2, y2)
        gc.line_to(x3, y3)

        gc.stroke_path()
        # test a single color channel.
        # note: This is a "screen capture" from running this
        #       test.  It looks right, but hasn't been checked closely.
        desired = array(
            [
                [255, 255, 255, 230, 255, 255, 255, 255, 255, 255],
                [255, 255, 231,  25, 212, 255, 255, 255, 255, 255],
                [255, 252,  65, 128, 255, 255, 255, 255, 255, 255],
                [255, 103,  26, 143, 229, 255, 255, 255, 255, 255],
                [180,   2, 118,  96,  23, 189, 255, 255, 205, 255],
                [255, 206, 255, 255, 189,  23,  97, 119,   2, 180],
                [255, 255, 255, 255, 255, 229, 142,  25, 103, 255],
                [255, 255, 255, 255, 255, 255, 127,  66, 252, 255],
                [255, 255, 255, 255, 255, 212,  26, 231, 255, 255],
                [255, 255, 255, 255, 255, 255, 231, 255, 255, 255],
            ]
        )
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)
