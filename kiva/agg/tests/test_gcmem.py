"""
Test for memory leak in the wx image backend.
"""

import unittest, sys
import gc as garbagecollector

from kiva.image import GraphicsContext, GraphicsContextSystem
from etsdevtools.debug.memusage import get_mem_usage

class test_agg(unittest.TestCase):
    def check_agg_mem_leak(self):
        pre = get_mem_usage()
        gc = GraphicsContext((500,500))
        del gc
        garbagecollector.collect()
        post = get_mem_usage()
        assert (pre == post)

    def check_wx_mem_leak(self):
        pre = get_mem_usage()
        gc = GraphicsContextSystem((500,500))
        del gc
        garbagecollector.collect()
        post = get_mem_usage()
        assert (pre == post)



def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(test_agg,'check_') )
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
