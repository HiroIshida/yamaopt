import os
import math
import numpy as np
from skrobot.coordinates.math import rotation_matrix
from numpy.lib.twodim_base import eye
from yamaopt.solver import KinematicSolver, SolverConfig
from yamaopt.polygon_constraint import polygon_to_trans_constraint
from data.sample_polygon import get_sample_real_polygons

np.random.seed(1)

def compute_numerical_jacobian(f, x0):
    f0 = f(x0)

    dim_inp = len(x0)

    eps =1e-6
    one_hots = [vec * eps for vec in np.eye(dim_inp)]

    is_out_jacobian = isinstance(f0, np.ndarray)


    if is_out_jacobian:
        dim_out = len(f0)
        jac = np.zeros((dim_out, dim_inp))
    else:
        jac = np.zeros(dim_inp) # actually gradient

    for i, dx in enumerate(one_hots):
        if is_out_jacobian:
            jac[:, i] = (f(x0 + dx) - f(x0)) / eps
        else:
            jac[i] = (f(x0 + dx) - f(x0)) / eps
    return jac

def test_compute_numerical_jacobian():
    f = lambda x: np.array([np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(x ** 2))])
    x0 = np.random.randn(3)
    jac_numel = compute_numerical_jacobian(f, x0)

    jac_real = np.array([x0 / np.sqrt(np.sum(x0 ** 2)), x0 / np.sqrt(np.sum(x0 ** 2))])
    np.testing.assert_almost_equal(jac_real, jac_numel, decimal=4)


def test_constraint():
    config_path = "./config/pr2_conf.yaml"
    config = SolverConfig.from_config_path(config_path)
    kinsol = KinematicSolver(config)

    def _test_constraint_jacobian(constraint, q_test):
        f = lambda q: constraint(q)[0]
        jac_numel = compute_numerical_jacobian(f, q_test)
        jac_anal = constraint(q_test)[1]
        np.testing.assert_almost_equal(jac_numel, jac_anal, decimal=4)

    polygon = np.array([[0.0, -0.3, -0.3], [0.0, 0.3, -0.3], [0.0, 0.3, 0.3], [0.0, -0.3, 0.3]])
    ineq_const, eq_const = kinsol.configuration_constraint_from_polygon(polygon, d_hover=0.0)

    q_tests = [np.random.randn(len(kinsol.control_joint_ids)) for _ in range(10)]
    # append case that fails with coarse discretization
    q_tests.append(np.array([-0.74996962, 2.0546241, 0.05340954, -0.4791571, 0.35016716,  0.01716473, -0.42914228])) 
    for q_test in q_tests:
        _test_constraint_jacobian(ineq_const, q_test)
        _test_constraint_jacobian(eq_const, q_test)

def test_objfun():
    config_path = "./config/pr2_conf.yaml"
    config = SolverConfig.from_config_path(config_path)
    kinsol = KinematicSolver(config)

    target_pos = np.ones(3)
    objfun = kinsol.create_objective_function(target_pos)

    def _test_objfun(objfun, q_test):
        f = lambda q: objfun(q)[0]
        grad_numel = compute_numerical_jacobian(f, q_test)
        grad_anal = objfun(q_test)[1]
        np.testing.assert_almost_equal(grad_numel, grad_anal, decimal=4)

    for _ in range(10):
        _test_objfun(objfun, np.random.randn(7))

def test_solve():
    config_path = "./config/pr2_conf.yaml"
    for use_base in [False, True]:
        config = SolverConfig.from_config_path(config_path, use_base)
        kinsol = KinematicSolver(config)

        polygon1 = np.array([[0.0, -0.3, -0.3], [0.0, 0.3, -0.3], [0.0, 0.3, 0.3], [0.0, -0.3, 0.3]])
        polygon1 += np.array([0.7, 0.0, 1.0])

        polygon2 = np.array([[0.5, -0.3, 0.0], [0.5, 0.3, 0.0], [0.0, 0.0, 0.6]])
        polygon2 += np.array([0.5, 0.0, 0.8])

        polygon3 = np.array([[0.7, 0.7, 0.0], [0.9, 0.5, 0.3], [0.5, 0.9, 0.3]])
        polygon3 += np.array([-0.4, -0.3, 0.9])

        d_hover = 0.00

        #for i, polygon in enumerate([polygon1, polygon2, polygon3]):
        for i, polygon in enumerate([polygon1, polygon2]):
            q_init = np.ones(7) * 0.3
            target_obj_pos = np.ones(3)

            sol = kinsol.solve(q_init, polygon, target_obj_pos, d_hover=d_hover)
            assert sol.success 
            assert len(sol.x) == 7 + 3 * use_base

            ineq, eq = kinsol.configuration_constraint_from_polygon(polygon, d_hover)
            P, jac = kinsol.forward_kinematics(sol.x)
            pos = P[:, :3]
            ineq, eq = polygon_to_trans_constraint(polygon, d_hover)
            assert ineq.is_satisfying(pos.flatten())
            assert eq.is_satisfying(pos.flatten())

def test_solve_multiple_with_artificial_data():
    config_path = "./config/pr2_conf.yaml"
    config = SolverConfig.from_config_path(config_path, use_base=True)
    kinsol = KinematicSolver(config)

    polygon1 = np.array([[1.0, -0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [1.0, -0.5, 0.5]]) + np.array([0, 0, 1.0])
    polygon2 = polygon1.dot(rotation_matrix(math.pi / 2.0, [0, 0, 1.0]).T)
    polygon3 = polygon1.dot(rotation_matrix(-math.pi / 2.0, [0, 0, 1.0]).T)

    polygons = [polygon1, polygon2, polygon3]
    q_init = np.ones(7) * 0.3

    target_obj_pos = np.array([-0.1, 0.7, 0.3])
    sol = kinsol.solve_multiple(q_init, polygons, target_obj_pos)
    np.testing.assert_equal(polygon2, sol.target_polygon)

    target_obj_pos = np.array([-0.1, -0.7, 0.3])
    sol = kinsol.solve_multiple(q_init, polygons, target_obj_pos)
    np.testing.assert_equal(polygon3, sol.target_polygon)

def test_solve_multiple_with_realistic_data():
    # Test using real polygon obtained from PR2 in the kitchen
    config_path = "./config/pr2_conf.yaml"
    config = SolverConfig.from_config_path(config_path, use_base=True)
    kinsol = KinematicSolver(config)

    polygons = get_sample_real_polygons()

    q_init = np.ones(7) * 0.3

    target_obj_pos = np.array([-0.1, 0.7, 0.3])
    d_hover = 0.0
    sol = kinsol.solve_multiple(q_init, polygons, target_obj_pos, d_hover=d_hover)
    pos, rpy = sol.end_coords[:3], sol.end_coords[3:]

    ineq, eq = polygon_to_trans_constraint(sol.target_polygon, d_hover)
    assert ineq.is_satisfying(pos)
    assert eq.is_satisfying(pos)
