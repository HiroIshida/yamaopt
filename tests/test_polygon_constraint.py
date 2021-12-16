import numpy as np
import math
from skrobot.coordinates.math import rpy2quaternion
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import rotation_matrix
from yamaopt.polygon_constraint import is_convex, polygon_to_trans_constraint
from yamaopt.polygon_constraint import polygon_to_desired_rpy

def simple_simplex():
    return np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])

def simple_square():
    return np.array([[0.0, -0.3, -0.3], [0.0, 0.3, -0.3], [0.0, 0.3, 0.3], [0.0, -0.3, 0.3]]) + np.array([1, 1, 1])

def simple_nonconvex():
    return np.array([[0, 0., 0.], [1., 0., 0.], [0.5, 0.5, 0.], [1., 1., 0], [0, 1., 0]])

def test_is_convex():
    assert is_convex(simple_simplex())
    assert is_convex(simple_square())
    assert not is_convex(simple_nonconvex())

def test_polygon_to_constraint():
    polygon = simple_simplex()
    Cineq, Ceq = polygon_to_trans_constraint(polygon)

    assert Ceq(np.random.randn(3)).shape == (1,)
    assert Cineq(np.random.randn(3)).shape == (3,)

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


    polygon = simple_square()
    Cineq, Ceq = polygon_to_trans_constraint(polygon)

    assert Ceq(np.random.randn(3)).shape == (1,)
    assert Cineq(np.random.randn(3)).shape == (4,)

    center = np.mean(polygon, axis=0)
    assert Ceq.is_satisfying(center)
    assert Cineq.is_satisfying(center)

    for pt in polygon:
        assert Ceq.is_satisfying(pt)
        assert Cineq.is_satisfying(pt)
        assert np.all(Cineq(pt) > -1e-7)

def test_polygon_to_desired_rpy():
    polygon1 = simple_simplex()
    polygon2 = simple_square()
    P = np.array([[1.0, -0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [1.0, -0.5, 0.5]]) + np.array([0, 0, 1.0])
    polygon3 = P.dot(rotation_matrix(math.pi * 0.7 / 2.0, [0, 0, 1.0]).T)
    polygon4 = P.dot(rotation_matrix(-math.pi / 2.0, [0, 0, 1.0]).T)

    for polygon in [polygon1, polygon2, polygon3, polygon4]:
        rpy = polygon_to_desired_rpy(polygon)
        q = rpy2quaternion(np.flip(rpy))
        matrix = quaternion2matrix(q)
        z_axis = matrix[:, 0]

        n_verts = len(polygon)
        for i in range(n_verts):
            p = polygon[i]
            q = polygon[0] if i==n_verts-1 else polygon[i+1]
            vec = p - q
            assert abs(z_axis.dot(vec)) < 1e-5
