from numpy import alltrue, ravel


class Utils(object):

    def assertRavelEqual(self, x, y):
        self.assert_(alltrue(ravel(x) == ravel(y)), "\n%s\n !=\n%s" % (x, y))
