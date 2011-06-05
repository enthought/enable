import unittest

from kiva.agg import GraphicsContextSystem

class GraphicsContextSystemTestCase(unittest.TestCase):

    def test_creation(self):
        """ Simply create and destroy multiple objects.  This silly
            test crashed when we transitioned from Numeric 23.1 to 23.8.
            That problem is fixed now.
        """
        for i in range(10):
            gc = GraphicsContextSystem((100,100), "rgba32")
            del gc

#----------------------------------------------------------------------------
# test setup code.
#----------------------------------------------------------------------------

def check_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(GraphicsContextSystemTestCase,'test_') )
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = check_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
