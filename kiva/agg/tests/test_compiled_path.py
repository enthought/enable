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

from numpy import array, pi

from kiva import agg

from .test_utils import Utils


class TestCompiledPath(unittest.TestCase, Utils):
    def test_init(self):
        agg.CompiledPath()

    def test_save_restore_ctm0(self):
        path = agg.CompiledPath()
        path.save_ctm()
        path.restore_ctm()

    def test_save_restore_ctm1(self):
        path = agg.CompiledPath()
        m0 = agg.translation_matrix(10.0, 10.0)
        path.set_ctm(m0)
        path.save_ctm()
        m1 = agg.translation_matrix(5.0, 5.0)
        path.set_ctm(m1)
        m2 = path.get_ctm()
        self.assertRavelEqual(m1, m2)
        path.restore_ctm()
        m3 = path.get_ctm()
        self.assertRavelEqual(m0, m3)

    def test_translate_ctm(self):
        path = agg.CompiledPath()
        path.translate_ctm(2.0, 2.0)
        actual = path.get_ctm()
        desired = agg.translation_matrix(2.0, 2.0)
        self.assertRavelEqual(actual, desired)

    def test_scale_ctm(self):
        path = agg.CompiledPath()
        path.scale_ctm(2.0, 2.0)
        actual = path.get_ctm()
        desired = agg.scaling_matrix(2.0, 2.0)
        self.assertRavelEqual(actual, desired)

    def test_rotate_ctm(self):
        angle = pi / 4.0
        path = agg.CompiledPath()
        path.rotate_ctm(angle)
        actual = path.get_ctm()
        desired = agg.rotation_matrix(angle)
        self.assertRavelEqual(actual, desired)

    def test_concat_ctm(self):
        path = agg.CompiledPath()
        path.translate_ctm(2.0, 2.0)
        m0 = agg.scaling_matrix(2.0, 2.0)
        path.concat_ctm(m0)
        actual = path.get_ctm()
        # wrapper not working
        # m0 *= agg.translation_matrix(2.0,2.0)
        m0.multiply(agg.translation_matrix(2.0, 2.0))
        desired = m0
        self.assertRavelEqual(actual, desired)

    def test_vertex(self):
        # !! should get this value from the agg enum value
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)
        actual, actual_flag = path._vertex()
        desired = array((1.0, 1.0))
        desired_flag = 1
        self.assertRavelEqual(actual, desired)
        self.assertRavelEqual(actual_flag, desired_flag)

        # check for end flag
        actual, actual_flag = path._vertex()
        desired_flag = 0
        self.assertRavelEqual(actual_flag, desired_flag)

    def test_vertices(self):
        # !! should get this value from the agg enum value
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)

        desired = array(((1.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
        actual = path._vertices()
        self.assertRavelEqual(actual, desired)

    def test_rewind(self):
        # !! should get this value from the agg enum value
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)
        actual, actual_flag = path._vertex()
        actual, actual_flag = path._vertex()

        path._rewind()
        actual, actual_flag = path._vertex()
        desired = array((1.0, 1.0))
        desired_flag = 1
        self.assertRavelEqual(actual, desired)
        self.assertRavelEqual(actual_flag, desired_flag)

    def test_begin_path(self):
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)
        path.begin_path()
        pt, flag = path._vertex()
        # !! should get this value from the agg enum value
        desired = 0
        self.assertRavelEqual(flag, desired)

    def test_move_to(self):
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)
        actual, flag = path._vertex()
        desired = array((1.0, 1.0))
        self.assertRavelEqual(actual, desired)

    def test_move_to1(self):
        """ Test that transforms are affecting move_to commands
        """
        path = agg.CompiledPath()
        path.translate_ctm(1.0, 1.0)
        path.move_to(1.0, 1.0)
        actual, flag = path._vertex()
        desired = array((2.0, 2.0))
        self.assertRavelEqual(actual, desired)

    def test_quad_curve_to(self):
        path = agg.CompiledPath()
        ctrl = 1.0, 1.0
        to = 2.0, 2.0
        path.quad_curve_to(ctrl[0], ctrl[1], to[0], to[1])
        actual_ctrl, flag = path._vertex()
        self.assertRavelEqual(actual_ctrl, ctrl)
        assert flag == 3
        actual_to, flag = path._vertex()
        assert actual_to == to
        assert flag == 3

    def test_curve_to(self):
        path = agg.CompiledPath()
        ctrl1 = 1.0, 1.0
        ctrl2 = 2.0, 2.0
        to = 3.0, 3.0
        path.curve_to(ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1], to[0], to[1])
        actual_ctrl1, flag = path._vertex()
        assert actual_ctrl1 == ctrl1
        assert flag == 4
        actual_ctrl2, flag = path._vertex()
        assert actual_ctrl2 == ctrl2
        assert flag == 4
        actual_to, flag = path._vertex()
        assert actual_to == to
        assert flag == 4

    def test_add_path(self):
        path1 = agg.CompiledPath()
        path1.move_to(1.0, 1.0)
        path1.translate_ctm(1.0, 1.0)
        path1.line_to(2.0, 2.0)  # actually (3.0,3.0)
        path1.scale_ctm(2.0, 2.0)
        path1.line_to(2.0, 2.0)  # actually (5.0,5.0)

        path2 = agg.CompiledPath()
        path2.move_to(1.0, 1.0)
        path2.translate_ctm(1.0, 1.0)
        path2.line_to(2.0, 2.0)  # actually (3.0,3.0)

        sub_path = agg.CompiledPath()
        sub_path.scale_ctm(2.0, 2.0)
        sub_path.line_to(2.0, 2.0)
        path2.add_path(sub_path)

        desired = path1._vertices()
        actual = path2._vertices()
        self.assertRavelEqual(actual, desired)

        desired = path1.get_ctm()
        actual = path2.get_ctm()
        self.assertRavelEqual(actual, desired)

    def base_helper_lines(self, lines):
        path = agg.CompiledPath()
        path.move_to(1.0, 1.0)
        path.line_to(2.0, 2.0)  # actually (3.0,3.0)
        path.lines(lines)
        actual = path._vertices()
        desired = array(
            (
                (1.0, 1.0, agg.path_cmd_move_to, agg.path_flags_none),
                (2.0, 2.0, agg.path_cmd_line_to, agg.path_flags_none),
                (3.0, 3.0, agg.path_cmd_move_to, agg.path_flags_none),
                (4.0, 4.0, agg.path_cmd_line_to, agg.path_flags_none),
                (0.0, 0.0, agg.path_cmd_stop, agg.path_flags_none),
            )
        )

        self.assertRavelEqual(actual, desired)

    def test_lines_array(self):
        lines = array(((3.0, 3.0), (4.0, 4.0)))
        self.base_helper_lines(lines)

    def test_lines_list(self):
        lines = [[3.0, 3.0], [4.0, 4.0]]
        self.base_helper_lines(lines)

    def test_rect(self):
        path = agg.CompiledPath()
        path.rect(1.0, 1.0, 1.0, 1.0)
        actual = path._vertices()
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
        self.assertRavelEqual(actual, desired)

    def base_helper_rects(self, rects):
        path = agg.CompiledPath()
        path.rects(rects)
        actual = path._vertices()
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
        self.assertRavelEqual(actual, desired)

    def test_rects_array(self):
        rects = array(((1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 1.0, 1.0)))
        self.base_helper_rects(rects)

    def test_rects_list(self):
        rects = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 1.0, 1.0]]
        self.base_helper_rects(rects)
