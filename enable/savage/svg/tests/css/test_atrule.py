import unittest

import enable.savage.svg.css.atrule as atrule


class TestAtKeyword(unittest.TestCase):
    def testValidKeywords(self):
        for kw in ["@import", "@page"]:
            self.assertEqual(kw, atrule.atkeyword.parseString(kw)[0])
