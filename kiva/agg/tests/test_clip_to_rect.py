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

    clip_to_rect() tests
    --------------------
        DONE *. clip_to_rect is inclusive on lower end and exclusive on upper
                end.
        DONE *. clip_to_rect behaves intelligently under scaled ctm.
        DONE *. clip_to_rect intersects input rect with the existing clipping
                rect.
        DONE *. current rectangular clipping path is saved/restored to the
                stack when save_state/restore_state are called.
        DONE *. clip_to_rect clears current path.
        DONE *. clip_to_rect raises NotImplementedError under a rotated ctm.

    clip_to_rects() tests
    ---------------------
        DONE *. Test that clip_to_rects raises not implemented, or whatever.

"""

import unittest

from numpy import array, transpose

from kiva.agg import GraphicsContextArray

from .test_utils import Utils


class ClipToRectTestCase(unittest.TestCase, Utils):

    # ------------------------------------------------------------------------
    # Simple Clipping to a single rectangle.
    # ------------------------------------------------------------------------

    def clip_to_rect_helper(self, desired, scale, clip_rects):
        """ desired -- 2D array with a single channels expected byte pattern.
            scale -- used in scale_ctm() to change the ctm.
            clip_args -- passed in as *clip_args to clip_to_rect.
        """
        shp = tuple(transpose(desired.shape))
        gc = GraphicsContextArray(shp, pix_format="rgb24")
        gc.scale_ctm(scale, scale)

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        if isinstance(clip_rects, tuple):
            gc.clip_to_rect(*clip_rects)
        else:
            for rect in clip_rects:
                gc.clip_to_rect(*rect)

        gc.rect(0, 0, 4, 4)

        # These settings allow the fastest path.
        gc.set_fill_color((0.0, 0.0, 0.0))  # black
        gc.fill_path()

        # test a single color channel
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_clip_to_rect_simple(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 2, 2)
        self.clip_to_rect_helper(desired, 1, clip_rect)

    def test_clip_to_rect_simple2(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [255,   0, 255, 255],
                [255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 1, 1)
        self.clip_to_rect_helper(desired, 1, clip_rect)

    def test_clip_to_rect_negative(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [  0,   0,   0, 255],  # noqa: E201
                [  0,   0,   0, 255],  # noqa: E201
                [  0,   0,   0, 255],  # noqa: E201
            ]
        )
        clip_rect = (-1, -1, 4, 4)
        self.clip_to_rect_helper(desired, 1, clip_rect)

    def test_clip_to_rect_simple3(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 2.49, 2.49)
        self.clip_to_rect_helper(desired, 1, clip_rect)

    def test_clip_to_rect_simple4(self):
        desired = array(
            [
                [255,   0,   0,   0],
                [255,   0,   0,   0],
                [255,   0,   0,   0],
                [255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 2.5, 2.5)
        self.clip_to_rect_helper(desired, 1, clip_rect)

    def test_clip_to_rect_simple5(self):
        # This tests clipping with a larger rectangle
        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )
        clip_rects = [(1, 1, 2, 2), (0, 0, 4, 4)]
        self.clip_to_rect_helper(desired, 1, clip_rects)

    def test_empty_clip_region(self):
        # This tests when the clipping region is clipped down to nothing.
        desired = array(
            [
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [255, 255, 255, 255],
                [255, 255, 255, 255],
            ]
        )
        clip_rects = [(1, 1, 4, 4), (3, 3, 1, 1), (1, 1, 1, 1)]
        self.clip_to_rect_helper(desired, 1, clip_rects)

    def test_clip_to_rect_scaled(self):
        desired = array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255,   0,   0,   0,   0, 255, 255],
                [255, 255,   0,   0,   0,   0, 255, 255],
                [255, 255,   0,   0,   0,   0, 255, 255],
                [255, 255,   0,   0,   0,   0, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 2, 2)
        self.clip_to_rect_helper(desired, 2.0, clip_rect)

    def test_clip_to_rect_scaled2(self):
        desired = array(
            [
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255,   0,   0,   0,   0,   0, 255],
                [255, 255,   0,   0,   0,   0,   0, 255],
                [255, 255,   0,   0,   0,   0,   0, 255],
                [255, 255,   0,   0,   0,   0,   0, 255],
                [255, 255,   0,   0,   0,   0,   0, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255, 255, 255, 255],
            ]
        )
        clip_rect = (1, 1, 2.25, 2.25)
        self.clip_to_rect_helper(desired, 2.0, clip_rect)

    def test_save_restore_clip_state(self):
        desired1 = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )
        desired2 = array(
            [
                [255,   0,   0,   0],
                [255,   0,   0,   0],
                [255,   0,   0,   0],
                [255, 255, 255, 255],
            ]
        )
        gc = GraphicsContextArray((4, 4), pix_format="rgb24")
        gc.clear((1.0, 1.0, 1.0))
        gc.set_fill_color((0.0, 0.0, 0.0))

        gc.clip_to_rect(1, 1, 3, 3)

        gc.save_state()
        gc.clip_to_rect(1, 1, 2, 2)
        gc.rect(0, 0, 4, 4)
        gc.fill_path()
        actual1 = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired1, actual1)
        gc.restore_state()

        gc.rect(0, 0, 4, 4)
        gc.fill_path()
        actual2 = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired2, actual2)

    @unittest.skip(
        "underlying library doesn't handleclipping to a rotated rectangle"
    )
    def test_clip_to_rect_rotated(self):
        # FIXME: test skipped
        #   This test raises an exception currently because the
        #   underlying library doesn't handle clipping to a rotated
        #   rectangle.  For now, we catch the the case with an
        #   exception, so that people can't screw up.  In the future,
        #   we should actually support this functionality.

        gc = GraphicsContextArray((1, 1), pix_format="rgb24")
        gc.rotate_ctm(1.0)

        self.assertRaises(NotImplementedError, gc.clip_to_rect, 0, 0, 1, 1)

    # ------------------------------------------------------------------------
    # Successive Clipping of multiple rectangles.
    # ------------------------------------------------------------------------

    def successive_clip_helper(self, desired, scale, clip_rect1, clip_rect2):
        """ desired -- 2D array with a single channels expected byte pattern.
            scale -- used in scale_ctm() to change the ctm.
            clip_rect1 -- 1st clipping path.
            clip_rect2 -- 2nd clipping path.
        """
        shp = tuple(transpose(desired.shape))
        gc = GraphicsContextArray(shp, pix_format="rgb24")
        gc.scale_ctm(scale, scale)

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        gc.clip_to_rect(*clip_rect1)
        gc.clip_to_rect(*clip_rect2)

        gc.rect(0, 0, 4, 4)

        # These settings allow the fastest path.
        gc.set_fill_color((0.0, 0.0, 0.0))  # black
        gc.fill_path()

        # test a single color channel
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_clip_successive_rects(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )

        clip_rect1 = (1, 1, 20, 20)
        clip_rect2 = (0, 0, 3, 3)

        self.successive_clip_helper(desired, 1.0, clip_rect1, clip_rect2)

    def test_clip_successive_rects2(self):
        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )

        clip_rect1 = (1, 1, 20, 20)
        clip_rect2 = (-1, -1, 4, 4)

        self.successive_clip_helper(desired, 1.0, clip_rect1, clip_rect2)

    # ------------------------------------------------------------------------
    # Save/Restore clipping path.
    # ------------------------------------------------------------------------

    def test_save_restore_clip_path(self):

        desired = array(
            [
                [255, 255, 255, 255],
                [255,   0,   0, 255],
                [255,   0,   0, 255],
                [255, 255, 255, 255],
            ]
        )

        # this is the clipping path we hope to see.
        clip_rect1 = (1, 1, 2, 2)

        # this will be a second path that will push/pop that should
        # never be seen.
        clip_rect2 = (1, 1, 1, 1)

        shp = tuple(transpose(desired.shape))
        gc = GraphicsContextArray(shp, pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        gc.clip_to_rect(*clip_rect1)

        # push and then pop a path that shouldn't affect the drawing
        gc.save_state()
        gc.clip_to_rect(*clip_rect2)
        gc.restore_state()

        gc.rect(0, 0, 4, 4)

        # These settings allow the fastest path.
        gc.set_fill_color((0.0, 0.0, 0.0))  # black
        gc.fill_path()

        # test a single color channel
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)

    def test_reset_path(self):
        """ clip_to_rect() should clear the current path.

            This is to maintain compatibility with the version
            of kiva that sits on top of Apple's Quartz engine.
        """
        desired = array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [255, 255, 0, 0],
            ]
        )

        shp = tuple(transpose(desired.shape))
        gc = GraphicsContextArray(shp, pix_format="rgb24")

        # clear background to white values (255, 255, 255)
        gc.clear((1.0, 1.0, 1.0))

        gc.rect(0, 0, 2, 4)

        gc.clip_to_rect(0, 0, 4, 4)
        gc.rect(2, 0, 2, 4)

        # These settings allow the fastest path.
        gc.set_fill_color((0.0, 0.0, 0.0))  # black
        gc.fill_path()

        # test a single color channel
        actual = gc.bmp_array[:, :, 0]
        self.assertRavelEqual(desired, actual)


class ClipToRectsTestCase(unittest.TestCase):
    def test_not_implemented(self):
        """ fix me: Currently not implemented, so we just ensure that
            any call to it throws an exception.
        """

        gc = GraphicsContextArray((1, 1), pix_format="rgb24")
        gc.rotate_ctm(1.0)

        # self.failUnlessRaises(
        #     NotImplementedError, gc.clip_to_rects, [[0, 0, 1, 1]]
        # )
