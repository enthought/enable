# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Test suite for affine transforms.

    :Author:      Eric Jones, Enthought, Inc., eric@enthought.com
    :Copyright:   Space Telescope Science Institute
    :License:     BSD Style

    So far, this is mainly a "smoke test" suite to make sure
    nothing is obviously wrong.
"""

import unittest

from numpy import alltrue, array, ravel

from kiva import affine
from kiva import basecore2d
from kiva import constants


class TestIsFullyTransparent(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(basecore2d.is_fully_transparent([1, 1, 1, 0]))
        self.assertTrue(not basecore2d.is_fully_transparent([0, 0, 0, 1]))
        self.assertTrue(not basecore2d.is_fully_transparent([0, 0, 0, 0.5]))


class TestFillEqual(unittest.TestCase):
    def test_simple(self):
        self.assertTrue(
            basecore2d.fill_equal(array([0, 0, 0, 0]), array([0, 0, 0, 0]))
        )
        self.assertTrue(
            not basecore2d.fill_equal(array([0, 0, 0, 0]), array([0, 0, 0, 1]))
        )
        self.assertTrue(
            not basecore2d.fill_equal(array([0, 0, 0, 0]), array([1, 0, 0, 0]))
        )


class LineStateTestCase(unittest.TestCase):
    def create_ls(self):
        color = array([0, 0, 0, 1])
        width = 2
        join = basecore2d.JOIN_MITER
        cap = basecore2d.CAP_ROUND
        phase = 0
        pattern = array([5, 5])
        dash = (phase, pattern)
        ls = basecore2d.LineState(color, width, cap, join, dash)
        return ls

    def test_create(self):
        self.create_ls()

    def test_color_on_copy(self):
        # The following test to make sure that a copy
        # was actually made of the line_color container(array).
        # If it isn't, both ls1 and ls2 will point at the same
        # data, and the change to the color affects both
        # line_states instead of just ls1.
        ls1 = self.create_ls()
        ls2 = ls1.copy()
        ls1.line_color[1] = 10
        self.assertTrue(not basecore2d.line_state_equal(ls1, ls2))

    def test_dash_on_copy(self):
        ls1 = self.create_ls()
        ls2 = ls1.copy()
        ls1.line_dash[1][0] = 10
        self.assertTrue(not basecore2d.line_state_equal(ls1, ls2))

    def test_cmp_for_different_length_dash_patterns(self):
        ls1 = self.create_ls()
        ls2 = ls1.copy()
        ls1.line_dash = (ls1.line_dash[0], array([10, 10, 10, 10]))
        self.assertTrue(not basecore2d.line_state_equal(ls1, ls2))

    def test_cmp(self):
        ls1 = self.create_ls()
        ls2 = ls1.copy()
        self.assertTrue(basecore2d.line_state_equal(ls1, ls2))

    # line_dash no longer allowed to be none.
    # def test_cmp_with_dash_as_none(self):
    #    ls1 = self.create_ls()
    #    ls2 = ls1.copy()
    #    #ls1.line_dash = None
    #    assert(not basecore2d.line_state_equal(ls1,ls2))


class GraphicsContextTestCase(unittest.TestCase):
    def test_create_gc(self):
        basecore2d.GraphicsContextBase()

    # ----------------------------------------------------------------
    # Test ctm transformations
    # ----------------------------------------------------------------

    def test_get_ctm(self):
        gc = basecore2d.GraphicsContextBase()
        # default ctm should be identity matrix.
        desired = affine.affine_identity()
        actual = gc.get_ctm()
        self.assertTrue(alltrue(ravel(actual == desired)))

    def test_scale_ctm(self):
        gc = basecore2d.GraphicsContextBase()
        ident = affine.affine_identity()
        sx, sy = 2.0, 3.0
        desired = affine.scale(ident, sx, sy)
        gc.scale_ctm(sx, sy)
        actual = gc.get_ctm()
        self.assertTrue(alltrue(ravel(actual == desired)))

    def test_rotate_ctm(self):
        gc = basecore2d.GraphicsContextBase()
        ident = affine.affine_identity()
        angle = 2.0
        desired = affine.rotate(ident, angle)
        gc.rotate_ctm(angle)
        actual = gc.get_ctm()
        self.assertTrue(alltrue(ravel(actual == desired)))

    def test_translate_ctm(self):
        gc = basecore2d.GraphicsContextBase()
        ident = affine.affine_identity()
        x, y = 2.0, 3.0
        desired = affine.translate(ident, x, y)
        gc.translate_ctm(x, y)
        actual = gc.get_ctm()
        self.assertTrue(alltrue(ravel(actual == desired)))

    def test_concat_ctm(self):
        gc = basecore2d.GraphicsContextBase()
        ident = affine.affine_identity()
        trans = affine.affine_from_rotation(2.0)
        desired = affine.concat(ident, trans)
        gc.concat_ctm(trans)
        actual = gc.get_ctm()
        self.assertTrue(alltrue(ravel(actual == desired)))

    # -------------------------------------------------------------------------
    # Setting drawing state variables
    #
    # These tests also check that the value is restored correctly with
    # save/restore state.
    #
    # Checks are only done to see if variables are represented correctly in
    # the graphics state.  The effects on graphics rendering for these
    # variables is not checked.  That is all pretty much handled in the
    # device_update_line_state and device_update_fill_state routines which
    # access the state variables.
    #
    # Note: The tests peek into the state object to see if the if state
    #       variables are set.  This ain't perfect, but core2d doesn't
    #       define accessor functions...
    # -------------------------------------------------------------------------

    def test_state_antialias(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to 1
        self.assertEqual(gc.state.antialias, 1)
        gc.set_antialias(0)
        gc.save_state()
        gc.set_antialias(1)
        self.assertEqual(gc.state.antialias, 1)
        gc.restore_state()
        self.assertEqual(gc.state.antialias, 0)

    def test_state_line_width(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to 1
        self.assertEqual(gc.state.line_state.line_width, 1)
        gc.set_line_width(5)
        gc.save_state()
        gc.set_line_width(10)
        self.assertEqual(gc.state.line_state.line_width, 10)
        gc.restore_state()
        self.assertEqual(gc.state.line_state.line_width, 5)

    def test_state_line_join(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to JOIN_MITER
        self.assertEqual(gc.state.line_state.line_join, constants.JOIN_MITER)
        gc.set_line_join(constants.JOIN_BEVEL)
        gc.save_state()
        gc.set_line_join(constants.JOIN_ROUND)
        self.assertEqual(gc.state.line_state.line_join, constants.JOIN_ROUND)
        gc.restore_state()
        self.assertEqual(gc.state.line_state.line_join, constants.JOIN_BEVEL)
        # set_line_join should fail if one attempts to set a bad value.
        self.assertRaises(ValueError, gc.set_line_join, (100,))

    def test_state_miter_limit(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to 1.0
        self.assertEqual(gc.state.miter_limit, 1.0)
        gc.set_miter_limit(2.0)
        gc.save_state()
        gc.set_miter_limit(3.0)
        self.assertEqual(gc.state.miter_limit, 3.0)
        gc.restore_state()
        self.assertEqual(gc.state.miter_limit, 2.0)

    def test_state_line_cap(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to CAP_ROUND
        self.assertEqual(gc.state.line_state.line_cap, constants.CAP_ROUND)
        gc.set_line_cap(constants.CAP_BUTT)
        gc.save_state()
        gc.set_line_cap(constants.CAP_SQUARE)
        self.assertEqual(gc.state.line_state.line_cap, constants.CAP_SQUARE)
        gc.restore_state()
        self.assertEqual(gc.state.line_state.line_cap, constants.CAP_BUTT)
        # set_line_cap should fail if one attempts to set a bad value.
        self.assertRaises(ValueError, gc.set_line_cap, (100,))

    def test_state_line_dash(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to non-dashed line
        self.assertTrue(not gc.state.line_state.is_dashed())
        gc.set_line_dash([1.0, 2.0], phase=2.0)
        gc.save_state()

        gc.set_line_dash([3.0, 4.0])
        self.assertTrue(gc.state.line_state.is_dashed())
        self.assertEqual(gc.state.line_state.line_dash[0], 0)
        self.assertTrue(
            alltrue(
                ravel(gc.state.line_state.line_dash[1] == array([3.0, 4.0]))
            )
        )
        gc.restore_state()
        self.assertTrue(gc.state.line_state.is_dashed())
        self.assertEqual(gc.state.line_state.line_dash[0], 2.0)
        self.assertTrue(
            alltrue(
                ravel(gc.state.line_state.line_dash[1] == array([1.0, 2.0]))
            )
        )

        # pattern must be a container with atleast two values
        self.assertRaises(ValueError, gc.set_line_cap, (100,))
        self.assertRaises(ValueError, gc.set_line_cap, ([100],))
        # phase must be positive.
        self.assertRaises(ValueError, gc.set_line_cap, ([100, 200], -1))

    def test_state_flatness(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to 1.0
        self.assertEqual(gc.state.flatness, 1.0)
        gc.set_flatness(2.0)
        gc.save_state()
        gc.set_flatness(3.0)
        self.assertEqual(gc.state.flatness, 3.0)
        gc.restore_state()
        self.assertEqual(gc.state.flatness, 2.0)

    def test_state_alpha(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to 1.0
        self.assertEqual(gc.state.alpha, 1.0)
        gc.set_alpha(0.0)
        gc.save_state()
        gc.set_alpha(0.5)
        self.assertEqual(gc.state.alpha, 0.5)
        gc.restore_state()
        self.assertEqual(gc.state.alpha, 0.0)

    def test_state_fill_color(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to [0,0,0,1]
        self.assertTrue(alltrue(gc.state.fill_color == array([0, 0, 0, 1])))
        gc.set_fill_color((0, 1, 0, 1))
        gc.save_state()
        gc.set_fill_color((1, 1, 1, 1))
        self.assertTrue(alltrue(gc.state.fill_color == array([1, 1, 1, 1])))
        gc.restore_state()
        self.assertTrue(alltrue(gc.state.fill_color == array([0, 1, 0, 1])))

    def test_state_stroke_color(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to [0,0,0,1]
        self.assertTrue(
            alltrue(gc.state.line_state.line_color == array([0, 0, 0, 1]))
        )
        gc.set_stroke_color((0, 1, 0, 1))
        gc.save_state()
        gc.set_stroke_color((1, 1, 1, 1))
        self.assertTrue(
            alltrue(gc.state.line_state.line_color == array([1, 1, 1, 1]))
        )
        gc.restore_state()
        self.assertTrue(
            alltrue(gc.state.line_state.line_color == array([0, 1, 0, 1]))
        )

    def test_state_character_spacing(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to None
        self.assertEqual(gc.state.character_spacing, 0.0)
        gc.set_character_spacing(1.0)
        gc.save_state()
        gc.set_character_spacing(2.0)
        self.assertEqual(gc.state.character_spacing, 2.0)
        gc.restore_state()
        self.assertEqual(gc.state.character_spacing, 1.0)

    def test_state_text_drawing_mode(self):
        gc = basecore2d.GraphicsContextBase()
        # defaults to None
        self.assertEqual(gc.state.text_drawing_mode, constants.TEXT_FILL)
        gc.set_text_drawing_mode(constants.TEXT_OUTLINE)
        gc.save_state()
        gc.set_text_drawing_mode(constants.TEXT_CLIP)
        self.assertEqual(gc.state.text_drawing_mode, constants.TEXT_CLIP)
        gc.restore_state()
        self.assertEqual(gc.state.text_drawing_mode, constants.TEXT_OUTLINE)
        # try an unacceptable value.
        self.assertRaises(ValueError, gc.set_text_drawing_mode, (10,))

    # -------------------------------------------------------------------------
    # Use context manager for saving and restoring state.
    # -------------------------------------------------------------------------

    def test_state_context_manager(self):
        gc = basecore2d.GraphicsContextBase()

        # Set an assortment of state properties.
        gc.set_antialias(0)
        gc.set_line_width(5)
        gc.set_fill_color((0, 1, 0, 1))

        with gc:
            # Change the state properties.
            gc.set_antialias(1)
            self.assertEqual(gc.state.antialias, 1)
            gc.set_line_width(10)
            self.assertEqual(gc.state.line_state.line_width, 10)
            gc.set_fill_color((1, 1, 1, 1))
            self.assertTrue(
                alltrue(gc.state.fill_color == array([1, 1, 1, 1]))
            )

        # Verify that we're back to the earlier settings.
        self.assertEqual(gc.state.antialias, 0)
        self.assertEqual(gc.state.line_state.line_width, 5)
        self.assertTrue(alltrue(gc.state.fill_color == array([0, 1, 0, 1])))

    def test_state_context_manager_nested(self):
        gc = basecore2d.GraphicsContextBase()

        # Set an assortment of state properties.
        gc.set_antialias(0)
        gc.set_line_width(5)
        gc.set_fill_color((0, 1, 0, 1))

        with gc:
            # Change the state properties.
            gc.set_antialias(1)
            self.assertEqual(gc.state.antialias, 1)
            gc.set_line_width(10)
            self.assertEqual(gc.state.line_state.line_width, 10)
            gc.set_fill_color((1, 1, 1, 1))
            self.assertTrue(
                alltrue(gc.state.fill_color == array([1, 1, 1, 1]))
            )

            with gc:
                # Change the state properties.
                gc.set_antialias(0)
                self.assertEqual(gc.state.antialias, 0)
                gc.set_line_width(2)
                self.assertEqual(gc.state.line_state.line_width, 2)
                gc.set_fill_color((1, 1, 0, 1))
                self.assertTrue(
                    alltrue(gc.state.fill_color == array([1, 1, 0, 1]))
                )

            # Verify that we're back to the earlier settings.
            self.assertEqual(gc.state.antialias, 1)
            self.assertEqual(gc.state.line_state.line_width, 10)
            self.assertTrue(
                alltrue(gc.state.fill_color == array([1, 1, 1, 1]))
            )

        # Verify that we're back to the earlier settings.
        self.assertEqual(gc.state.antialias, 0)
        self.assertEqual(gc.state.line_state.line_width, 5)
        self.assertTrue(alltrue(gc.state.fill_color == array([0, 1, 0, 1])))

    # -------------------------------------------------------------------------
    # Begin/End Page
    # These are implemented yet.  The tests are just here to remind me that
    # they need to be.
    # -------------------------------------------------------------------------

    def test_begin_page(self):
        # just to let me know it needs implementation.
        gc = basecore2d.GraphicsContextBase()
        gc.begin_page()

    def test_end_page(self):
        # just to let me know it needs implementation.
        gc = basecore2d.GraphicsContextBase()
        gc.end_page()

    # -------------------------------------------------------------------------
    # flush/synchronize
    # These are implemented yet.  The tests are just here to remind me that
    # they need to be.
    # -------------------------------------------------------------------------

    def test_synchronize(self):
        # just to let me know it needs implementation.
        gc = basecore2d.GraphicsContextBase()
        gc.synchronize()

    def test_flush(self):
        # just to let me know it needs implementation.
        gc = basecore2d.GraphicsContextBase()
        gc.flush()

    # -------------------------------------------------------------------------
    # save/restore state.
    #
    # Note: These test peek into the state object to see if the if state
    #       variables are set.  This ain't perfect, but core2d doesn't
    #       define accessor functions...
    #
    # items that need to be tested:
    #   ctm
    #   clip region (not implemented)
    #   line width
    #   line join
    #
    # -------------------------------------------------------------------------

    def test_save_state_line_width(self):
        gc = basecore2d.GraphicsContextBase()
        gc.set_line_width(5)
        gc.save_state()
        gc.set_line_width(10)
        self.assertEqual(gc.state.line_state.line_width, 10)
        gc.restore_state()
        self.assertEqual(gc.state.line_state.line_width, 5)

    # -------------------------------------------------------------------------
    # Test drawing path empty
    # -------------------------------------------------------------------------

    def test_is_path_empty1(self):
        """ A graphics context should start with an empty path.
        """
        gc = basecore2d.GraphicsContextBase()
        self.assertTrue(gc.is_path_empty())

    def test_is_path_empty2(self):
        """ A path that has moved to a point, but still hasn't drawn
            anything is empty.
        """
        gc = basecore2d.GraphicsContextBase()
        x, y = 1.0, 2.0
        gc.move_to(x, y)
        self.assertTrue(gc.is_path_empty())

    def test_is_path_empty3(self):
        """ A path that has moved to a point multiple times, but hasn't drawn
            anything is empty.
        """
        gc = basecore2d.GraphicsContextBase()
        x, y = 1.0, 2.0
        gc.move_to(x, y)
        # this should create another path.
        x, y = 1.0, 2.5
        gc.move_to(x, y)
        self.assertTrue(gc.is_path_empty())

    def test_is_path_empty4(self):
        """ We've added a line, so the path is no longer empty.
        """
        gc = basecore2d.GraphicsContextBase()
        x, y = 1.0, 2.0
        gc.move_to(x, y)
        gc.line_to(x, y)
        self.assertTrue(not gc.is_path_empty())
