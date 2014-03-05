from enable.gadgets.ctf.piecewise import PiecewiseFunction, verify_values


def test_piecewise_insert():
    pf = PiecewiseFunction(key=lambda x: x[0])

    values = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    for val in values:
        pf.insert(val)

    assert pf.items() == values
    assert pf.value_at(1) == values[1]
    assert pf.items() == pf.values()


def test_piecewise_neighbors():
    pf = PiecewiseFunction(key=lambda x: x[0])

    values = [(0.0, 0.0), (0.5, 0.5), (0.9, 1.0)]
    for val in values:
        pf.insert(val)

    assert pf.neighbor_indices(0.5) == (1, 2)
    assert pf.neighbor_indices(1.0) == (2, 2)


def test_piecewise_remove():
    pf = PiecewiseFunction(key=lambda x: x[0])

    values = [(0.0, 0.0), (0.5, 0.5), (0.9, 1.0)]
    for val in values:
        pf.insert(val)

    pf.remove(0)
    assert pf.items() == values[1:]


def test_piecewise_update():
    pf = PiecewiseFunction(key=lambda x: x[0])

    values = [(0.0, 0.0), (0.5, 0.5), (0.9, 1.0)]
    for val in values:
        pf.insert(val)

    values[0] = (0.0, 0.5)
    pf.update(0, values[0])
    assert pf.items() == values


def test_verify_values():
    good_values = [(0.0, 0.0), (1.0, 1.0)]
    short_subvalues = [(0.0,), (1.0,)]
    non_uniform_subvalues = [(0.0, 1.0), (1.0, 1.0, 1.0)]
    int_subvalues = [(0, 1), (1, 0)]
    out_of_range_subvalues = [(0.0, 1.1), (1.0, 2.0)]
    bad_data = range(10)

    assert verify_values(good_values)
    assert not verify_values(short_subvalues)
    assert not verify_values(non_uniform_subvalues)
    assert not verify_values(int_subvalues)
    assert not verify_values(out_of_range_subvalues)
    assert not verify_values(bad_data)
