import unittest
import pdb

from enthought.enable2.api import Component, Container, Viewport


class ViewportTestCase(unittest.TestCase):

    def test_basic_viewport(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(component=container, view_position=[10.0, 10.0],
                        view_bounds=[50.0, 50.0])
        #pdb.set_trace()
        self.assert_(view.components_at(0.0, 0.0)[0] == component)
        self.assert_(view.components_at(44.9, 0.0)[0] == component)
        self.assert_(view.components_at(0.0, 44.9)[0] == component)
        self.assert_(view.components_at(44.9, 44.9)[0] == component)
        
        self.assert_(view.components_at(46.0, 45.0) == [])
        self.assert_(view.components_at(46.0, 0.0) == [])
        self.assert_(view.components_at(45.0, 46.0) == [])
        self.assert_(view.components_at(0.0, 46.0) == [])
        return


def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append(unittest.makeSuite(ViewportTestCase, 'test_'))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
    
# EOF
