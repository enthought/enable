""" Tests for kiva.fonttools.font
"""
import os
import unittest

from kiva.fonttools import Font
from kiva.fonttools.tests._testing import patch_global_font_manager


class TestFont(unittest.TestCase):
    def setUp(self):
        # Invalidate the global font manager cache to avoid test interaction
        # as well as catching erroneous assumption on an existing cache.
        font_manager_patcher = patch_global_font_manager(None)
        font_manager_patcher.start()
        self.addCleanup(font_manager_patcher.stop)

    def test_find_font_empty_name(self):
        # This test relies on the fact there exists some fonts on the system
        # that the font manager can load. Ideally we should be able to redirect
        # the path from which the font manager loads font files, then this test
        # can be less fragile.
        font = Font(face_name="")
        font_file_path = font.findfont()
        self.assertTrue(os.path.exists(font_file_path))

    def test_find_font_some_face_name(self):
        font = Font(face_name="ProbablyNotFound")

        # There will be warnings as there will be no match for the requested
        # face name.
        with self.assertWarns(UserWarning):
            font_file_path = font.findfont()
        self.assertTrue(os.path.exists(font_file_path))

    def test_find_font_name(self):
        font = Font(face_name="ProbablyNotFound")

        # There will be warnings as there will be no match for the requested
        # face name.
        with self.assertWarns(UserWarning):
            name = font.findfontname()

        # Name should be nonempty.
        self.assertGreater(len(name), 0)
