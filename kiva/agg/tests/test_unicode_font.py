import unittest

from kiva.agg import AggFontType, GraphicsContextArray
from kiva.fonttools import Font


class UnicodeTest(unittest.TestCase):
    def test_show_text_at_point(self):
        gc = GraphicsContextArray((100, 100))
        gc.set_font(Font())
        gc.show_text_at_point(str("asdf"), 5, 5)

    def test_agg_font_type(self):
        f1 = AggFontType("Arial")
        f2 = AggFontType(b"Arial")
        self.assertEqual(f1, f2)


if __name__ == "__main__":
    unittest.main()
