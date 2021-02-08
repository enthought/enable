# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from contextlib import contextmanager
import os
import tempfile
import unittest

from kiva.pdfmetrics import parseAFMFile


@contextmanager
def ensure_deleted():
    file_handle = None
    try:
        file_handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
        yield file_handle
    finally:
        if file_handle is not None:
            os.unlink(file_handle.name)


class TestparseAFMFile(unittest.TestCase):
    def test_empty_file(self):
        # Empty file
        with ensure_deleted() as f:
            f.close()
            with self.assertRaises(ValueError):
                parseAFMFile(f.name)

    def test_single_line_file(self):
        # Single Line
        with ensure_deleted() as f:
            f.write("SINGLE LINE")
            f.close()
            with self.assertRaises(ValueError):
                parseAFMFile(f.name)

    def test_two_lines_file(self):
        # Single Line
        text = "SINGLE LINE\nSECOND LINE"

        with ensure_deleted() as f:
            f.write(text)
            f.close()
            topLevel, glyphLevel = parseAFMFile(f.name)
            self.assertEqual(topLevel, {})
            self.assertEqual(glyphLevel, [])

        text_mac = text.replace("\n", "\r")
        with ensure_deleted() as f:
            f.write(text_mac)
            f.close()
            topLevel, glyphLevel = parseAFMFile(f.name)
            self.assertEqual(topLevel, {})
            self.assertEqual(glyphLevel, [])
