import numpy as np
from yamaopt.polygon_constraint import is_convex, polygon_to_constraint

def simple_simplex():
    return np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])

def simple_nonconvex():
    return np.array([[0, 0., 0.], [1., 0., 0.], [0.5, 0.5, 0.], [1., 1., 0], [0, 1., 0]])

def test_is_convex():
    assert is_convex(simple_simplex())
    assert not is_convex(simple_nonconvex())

def test_polygon_to_constraint():
    polygon = simple_simplex()
    Ceq, Cineq = polygon_to_constraint(polygon)

    center = np.mean(polygon, axis=0)
    assert Ceq.is_satisfying(center)
    assert Cineq.is_satisfying(center)

    for pt in polygon:
        assert Ceq.is_satisfying(pt)
        assert Cineq.is_satisfying(pt)
        assert np.all(Cineq(pt) > -1e-7)

    for pt in [np.zeros(3), np.ones(3), -np.ones(3)]:
        assert not Ceq.is_satisfying(pt)
        assert Cineq.is_satisfying(pt)

    for pt in [np.array([0, 0, 2.]), np.array([0, 2., 0.])]:
        assert not Ceq.is_satisfying(pt)
        assert not Cineq.is_satisfying(pt)
