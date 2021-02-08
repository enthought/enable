# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest

from numpy import all, allclose, array, dtype, pi, ones

from kiva import agg
from kiva.api import Font


class GraphicsContextArrayTestCase(unittest.TestCase):
    def test_init(self):
        agg.GraphicsContextArray((100, 100))

    def test_init_bmp_equal_to_clear_bmp(self):
        gc = agg.GraphicsContextArray((5, 5))
        gc2 = agg.GraphicsContextArray((5, 5))
        gc2.clear()
        self.assertTrue((gc.bmp_array == gc2.bmp_array).all())

    def test_init_with_bmp_doesnt_clear(self):
        a = ones((5, 5, 4), dtype("uint8"))
        gc = agg.GraphicsContextArray(a, pix_format="rgba32")
        self.assertTrue((gc.bmp_array == a).all())

    def test_save_restore_state(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.save_state()
        gc.restore_state()

    def test_save_restore_state_for_ctm(self):
        gc = agg.GraphicsContextArray((100, 100))
        m0 = agg.translation_matrix(10.0, 10.0)
        gc.set_ctm(m0)
        gc.save_state()
        m1 = agg.translation_matrix(5.0, 5.0)
        gc.set_ctm(m1)
        m2 = gc.get_ctm()
        self.assertEqual(tuple(m1), m2)
        gc.restore_state()
        m3 = gc.get_ctm()
        self.assertEqual(tuple(m0), m3)

    # !! Need some tests of other graphics state information on
    # !! save/restore state

    def test_save_restore_state_for_ttm(self):
        # The interesting thing here is that we are verifying
        # that the text transform matrix (TTM) is *not* saved
        # with the graphics state.
        gc = agg.GraphicsContextArray((100, 100))
        m0 = agg.translation_matrix(10.0, 10.0)
        gc.set_text_matrix(m0)
        gc.save_state()
        gc.set_text_matrix(agg.translation_matrix(5.0, 5.0))
        gc.restore_state()
        m1 = gc.get_text_matrix()
        self.assertNotEqual(m1, m0)

    # !! Need some tests of other graphics state information on
    # !! save/restore state

    def test_context_manager(self):
        gc = agg.GraphicsContextArray((100, 100))

        # Set some values.
        gc.set_stroke_color((1, 0, 0, 1))
        gc.set_antialias(0)
        gc.set_alpha(0.25)

        with gc:
            # Change the values in the current context.
            gc.set_stroke_color((0, 0, 1, 1))
            self.assertTrue(all(gc.get_stroke_color() == (0, 0, 1, 1)))
            gc.set_antialias(1)
            self.assertEqual(gc.get_antialias(), 1)
            gc.set_alpha(0.75)
            self.assertEqual(gc.get_alpha(), 0.75)

        # Verify that we are back to the previous settings.
        self.assertTrue(all(gc.get_stroke_color() == (1, 0, 0, 1)))
        self.assertEqual(gc.get_antialias(), 0)
        self.assertEqual(gc.get_alpha(), 0.25)

    def test_context_manager_nested(self):
        gc = agg.GraphicsContextArray((100, 100))

        # Set some values.
        gc.set_stroke_color((1, 0, 0, 1))
        gc.set_antialias(0)
        gc.set_alpha(0.25)

        with gc:
            # Change the values in the current context.
            gc.set_stroke_color((0, 0, 1, 1))
            self.assertTrue(all(gc.get_stroke_color() == (0, 0, 1, 1)))
            gc.set_antialias(1)
            self.assertEqual(gc.get_antialias(), 1)
            gc.set_alpha(0.75)
            self.assertEqual(gc.get_alpha(), 0.75)

            with gc:
                # Change the values in the current context.
                gc.set_stroke_color((1, 0, 1, 1))
                self.assertTrue(all(gc.get_stroke_color() == (1, 0, 1, 1)))
                gc.set_antialias(0)
                self.assertEqual(gc.get_antialias(), 0)
                gc.set_alpha(1.0)
                self.assertEqual(gc.get_alpha(), 1.0)

            # Verify that we are back to the previous settings.
            self.assertTrue(all(gc.get_stroke_color() == (0, 0, 1, 1)))
            self.assertEqual(gc.get_antialias(), 1)
            self.assertEqual(gc.get_alpha(), 0.75)

        # Verify that we are back to the previous settings.
        self.assertTrue(all(gc.get_stroke_color() == (1, 0, 0, 1)))
        self.assertEqual(gc.get_antialias(), 0)
        self.assertEqual(gc.get_alpha(), 0.25)

    def test_translate_ctm(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.translate_ctm(2.0, 2.0)
        actual = gc.get_ctm()
        desired = agg.translation_matrix(2.0, 2.0)
        self.assertEqual(actual, tuple(desired))

    def test_scale_ctm(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.scale_ctm(2.0, 2.0)
        actual = gc.get_ctm()
        desired = agg.scaling_matrix(2.0, 2.0)
        self.assertEqual(actual, tuple(desired))

    def test_rotate_ctm(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.rotate_ctm(pi / 4.0)
        actual = gc.get_ctm()
        desired = agg.rotation_matrix(pi / 4.0)
        self.assertEqual(actual, tuple(desired))

    def test_concat_ctm(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.translate_ctm(2.0, 2.0)
        m0 = agg.scaling_matrix(2.0, 2.0)
        gc.concat_ctm(m0)
        actual = gc.get_ctm()
        m0.multiply(agg.translation_matrix(2.0, 2.0))
        desired = m0
        self.assertEqual(actual, tuple(desired))

    def test_begin_path(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.move_to(1.0, 1.0)
        gc.begin_path()
        path = gc._get_path()
        pt, flag = path._vertex()
        # !! should get this value from the agg enum value
        desired = 0
        self.assertEqual(flag, desired)

    def test_move_to(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.move_to(1.0, 1.0)
        path = gc._get_path()
        actual, flag = path._vertex()
        desired = array((1.0, 1.0))
        self.assertTrue(allclose(actual, desired))

    def test_move_to1(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.translate_ctm(1.0, 1.0)
        gc.move_to(1.0, 1.0)
        path = gc._get_path()
        actual, flag = path._vertex()
        desired = array((2.0, 2.0))
        self.assertTrue(allclose(actual, desired))

    def test_quad_curve_to(self):
        gc = agg.GraphicsContextArray((100, 100))
        ctrl = 1.0, 1.0
        to = 2.0, 2.0
        gc.quad_curve_to(ctrl[0], ctrl[1], to[0], to[1])
        path = gc._get_path()
        actual_ctrl, flag = path._vertex()
        self.assertEqual(actual_ctrl, ctrl)
        self.assertEqual(flag, 3)
        actual_to, flag = path._vertex()
        self.assertEqual(actual_to, to)
        self.assertEqual(flag, 3)

    def test_curve_to(self):
        gc = agg.GraphicsContextArray((100, 100))
        ctrl1 = 1.0, 1.0
        ctrl2 = 2.0, 2.0
        to = 3.0, 3.0
        gc.curve_to(ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1], to[0], to[1])

        path = gc._get_path()
        actual_ctrl1, flag = path._vertex()
        self.assertEqual(actual_ctrl1, ctrl1)
        self.assertEqual(flag, 4)
        actual_ctrl2, flag = path._vertex()
        self.assertEqual(actual_ctrl2, ctrl2)
        self.assertEqual(flag, 4)
        actual_to, flag = path._vertex()
        self.assertEqual(actual_to, to)
        self.assertEqual(flag, 4)

    def test_add_path(self):
        path1 = agg.CompiledPath()
        path1.move_to(1.0, 1.0)
        path1.translate_ctm(1.0, 1.0)
        path1.line_to(2.0, 2.0)  # actually (3.0,3.0)
        path1.scale_ctm(2.0, 2.0)
        path1.line_to(2.0, 2.0)  # actually (5.0,5.0)

        gc = agg.GraphicsContextArray((100, 100))
        gc.move_to(1.0, 1.0)
        gc.translate_ctm(1.0, 1.0)
        gc.line_to(2.0, 2.0)  # actually (3.0,3.0)

        sub_path = agg.CompiledPath()
        sub_path.scale_ctm(2.0, 2.0)
        sub_path.line_to(2.0, 2.0)
        gc.add_path(sub_path)

        path2 = gc._get_path()
        desired = path1._vertices()
        actual = path2._vertices()
        self.assertTrue(allclose(actual, desired))

        desired = path1.get_ctm()
        actual = path2.get_ctm()
        self.assertEqual(actual, desired)

    def base_lines(self, lines):
        gc = agg.GraphicsContextArray((100, 100))
        gc.move_to(1.0, 1.0)
        gc.line_to(2.0, 2.0)  # actually (3.0,3.0)
        gc.lines(lines)
        actual = gc._get_path()._vertices()
        desired = array(
            (
                (1.0, 1.0, agg.path_cmd_move_to, agg.path_flags_none),
                (2.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (3.0, 3.0, agg.path_cmd_move_to, agg.path_flags_none),
                (4.0, 4.0, agg.path_cmd_line_to, agg.path_flags_none),
                (0.0, 0.0, agg.path_cmd_stop, agg.path_flags_none),
            )
        )

        self.assertTrue(allclose(actual, desired))

    def test_lines_array(self):
        lines = array(((3.0, 3.0), (4.0, 4.0)))
        self.base_lines(lines)

    def test_lines_list(self):
        lines = [[3.0, 3.0], [4.0, 4.0]]
        self.base_lines(lines)

    def base_rects(self, rects):
        gc = agg.GraphicsContextArray((100, 100))
        gc.rects(rects)
        actual = gc._get_path()._vertices()
        desired = array(
            (
                (1.0, 1.0, agg.path_cmd_move_to, agg.path_flags_none),
                (1.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (2.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (2.0, 1.0, agg.path_cmd_line_to, agg.path_flags_none),
                (0.0, 0.0, agg.path_cmd_end_poly, agg.path_flags_close),
                (2.0, 2.0, agg.path_cmd_move_to, agg.path_flags_none),
                (2.0, 3.0, agg.path_cmd_line_to, agg.path_flags_none),
                (3.0, 3.0, agg.path_cmd_line_to, agg.path_flags_none),
                (3.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (0.0, 0.0, agg.path_cmd_end_poly, agg.path_flags_close),
                (0.0, 0.0, agg.path_cmd_stop, agg.path_flags_none),
            )
        )
        self.assertTrue(allclose(actual, desired))

    def test_rects_array(self):
        rects = array(((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 1.0, 1.0)))
        self.base_rects(rects)

    def test_rects_list(self):
        rects = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]
        self.base_rects(rects)

    def test_rect(self):
        gc = agg.GraphicsContextArray((100, 100))
        gc.rect(1.0, 1.0, 1.0, 1.0)
        actual = gc._get_path()._vertices()
        desired = array(
            (
                (1.0, 1.0, agg.path_cmd_move_to, agg.path_flags_none),
                (1.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (2.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (2.0, 1.0, agg.path_cmd_line_to, agg.path_flags_none),
                (0.0, 0.0, agg.path_cmd_end_poly, agg.path_flags_close),
                (0.0, 0.0, agg.path_cmd_stop, agg.path_flags_none),
            )
        )
        self.assertTrue(allclose(actual, desired))

    def test_clip_to_rect(self):
        gc = agg.GraphicsContextArray((10, 10))
        gc.move_to(0, 0)
        gc.line_to(10, 10)
        gc.clip_to_rect(5, 5, 5, 5)
        gc.stroke_path()
        # make sure nothing was drawn in the corner
        self.assertEqual(gc.bmp_array[-1, 0, 0], 255)

    def test_stroke_path(self):
        gc = agg.GraphicsContextArray((5, 5))
        gc.move_to(0, 0)
        gc.line_to(5, 5)
        gc.stroke_path()
        # assert the lower left and upper corner are the same,
        # and have something drawn in them.
        self.assertEqual(gc.bmp_array[-1, 0, 0], gc.bmp_array[0, -1, 0])
        self.assertNotEqual(gc.bmp_array[-1, 0, 0], 255)

    def test_set_get_text_position(self):
        gc = agg.GraphicsContextArray((5, 5))
        gc.set_text_position(1, 1)
        actual = gc.get_text_position()
        desired = (1, 1)
        self.assertTrue(allclose(actual, desired))

    def test_get_set_font(self):
        gc = agg.GraphicsContextArray((5, 5))
        font1 = Font("modern")
        gc.set_font(font1)
        font3 = gc.get_font()
        self.assertEqual(font1.face_name, font3.name)
        self.assertEqual(font1.size, font3.size)
        self.assertEqual(font1.family, font3.family)
        self.assertEqual(font1.style, font3.style)
        self.assertEqual(font1.encoding, font3.encoding)

    def test_set_line_dash_none(self):
        gc = agg.GraphicsContextArray((5, 5))
        gc.set_line_dash(None)
        # !! need to add an accessor to test result

    def test_set_line_dash_list(self):
        gc = agg.GraphicsContextArray((5, 5))
        gc.set_line_dash([2, 3])
        # !! need to add an accessor to test result

    def test_set_line_dash_2d_list(self):
        gc = agg.GraphicsContextArray((5, 5))
        try:
            gc.set_line_dash([[2, 3], [2, 3]])
        except TypeError:
            pass

    def test_set_text_matrix_ndarray(self):
        """ Test that gc.set_text_matrix accepts 3x3 ndarrays. """
        gc = agg.GraphicsContextArray((5, 5))
        m = array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 1.0]])
        gc.set_text_matrix(m)
        m2 = gc.get_text_matrix()
        self.assertEqual(m2, agg.AffineMatrix(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
