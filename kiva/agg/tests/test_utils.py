from numpy import alltrue, ravel


class Utils(object):

    def assertRavelEqual(self, x, y):
        try:
            self.assert_(alltrue(ravel(x) == ravel(y)), "\n%s\n !=\n%s" % (x, y))
        except ValueError:
            print "x:", x
            print "y:", y
            raise
