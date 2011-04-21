import unittest
from enable.savage.svg.css import block

class TestBlockParsing(unittest.TestCase):
    def testBlock(self):
        """ Not a valid CSS statement, but a valid block
            This tests some abuses of quoting and escaping
            See http://www.w3.org/TR/REC-CSS2/syndata.html Section 4.1.6
        """
        self.assertEqual(
            [["causta:", '"}"', "+", "(", ["7"], "*", "'\\''", ")"]],
            block.block.parseString(r"""{ causta: "}" + ({7} * '\'') }""").asList()
        )
