import unittest
from kiva.agg import AggFontType, GraphicsContextArray
from kiva.fonttools import Font

class UnicodeTest(unittest.TestCase):


    def test_show_text_at_point(self):
        gc = GraphicsContextArray((100,100))
        gc.set_font(Font())
        gc.show_text_at_point(unicode('asdf'), 5,5)

    def test_agg_font_type(self):
        f = AggFontType(u"Arial")

if __name__ == "__main__":
    unittest.main()
