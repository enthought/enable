from enthought.util.numerix import alltrue, ravel


def assert_arrays_equal(desired, actual):    
    """ Compare to arrays and assert that they are equal
    """
    assert alltrue(ravel(desired) == ravel(actual)), \
           "desired!= actual:\n%s \n!=\n %s" % (desired, actual)
