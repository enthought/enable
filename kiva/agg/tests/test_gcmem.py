"""
Test for memory leak in the wx image backend.
"""

import unittest, sys
import gc as garbagecollector

from kiva.image import GraphicsContext, GraphicsContextSystem
try:
    from etsdevtools.debug.memusage import get_mem_usage
    etsdevtools_available = True
except ModuleNotFoundError:
    etsdevtools_available = False


@unittest.skipIf(not etsdevtools_available, "test requires etsdevtools")
class test_agg(unittest.TestCase):
    def test_agg_mem_leak(self):
        pre = get_mem_usage()
        gc = GraphicsContext((500,500))
        del gc
        garbagecollector.collect()
        post = get_mem_usage()
        assert (pre == post)

    def test_wx_mem_leak(self):
        pre = get_mem_usage()
        gc = GraphicsContextSystem((500,500))
        del gc
        garbagecollector.collect()
        post = get_mem_usage()
        assert (pre == post)
