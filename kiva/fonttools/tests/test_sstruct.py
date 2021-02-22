# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from unittest import TestCase

from ..sstruct import SStructError, calcsize, getformat, pack, unpack


class TestSStruct(TestCase):
    def test_roundtrip(self):
        format = """
            # comments are allowed
            >  # big endian (see documentation for struct)
            # empty lines are allowed:

            ashort: h
            along: l
            abyte: b    # a byte
            achar: c
            astr: 5s
            afloat: f; adouble: d   # multiple "statements" are allowed
            afixed: 16.16F
        """
        self.assertEqual(calcsize(format), 29)

        class foo(object):
            pass

        i = foo()

        i.ashort = 0x7FFF
        i.along = 0x7FFFFFFF
        i.abyte = 0x7F
        i.achar = b"a"
        i.astr = b"12345"
        i.afloat = 0.5
        i.adouble = 0.5
        i.afixed = 1.5

        data = pack(format, i)
        self.assertEqual(
            data,
            b"\x7f\xff"
            + b"\x7f\xff\xff\xff"
            + b"\x7f"
            + b"a"
            + b"12345"
            + b"\x3f\x00\x00\x00"
            + b"\x3f\xe0\x00\x00\x00\x00\x00\x00"
            + b"\x00\x01\x80\x00",
        )

        self.assertEqual(
            unpack(format, data),
            {
                "ashort": i.ashort,
                "abyte": i.abyte,
                "achar": i.achar,
                "along": i.along,
                "astr": i.astr,
                "afloat": i.afloat,
                "adouble": i.adouble,
                "afixed": i.afixed,
            },
        )

        i2 = foo()
        unpack(format, data, i2)
        self.assertEqual(vars(i), vars(i2))

    def test_bad_format_char(self):
        format = "test: b; >"
        with self.assertRaises(SStructError):
            getformat(format)

    def test_format_syntax_error(self):
        format = "test: z"
        with self.assertRaises(SStructError):
            getformat(format)

    def test_fixed_point_error(self):
        format = "test: 4.8F"
        with self.assertRaises(SStructError):
            getformat(format)
