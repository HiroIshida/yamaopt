import os
import math
import numpy as np
from skrobot.coordinates.math import rotation_matrix
from numpy.lib.twodim_base import eye
from yamaopt.solver import KinematicSolver, SolverConfig
from yamaopt.polygon_constraint import polygon_to_trans_constraint

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
    ineq_const, eq_const = kinsol.configuration_constraint_from_polygon(polygon)
    for _ in range(10):
        q_test = np.random.randn(len(kinsol.control_joint_ids))
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
        polygon2 += np.array([0.4, 0.0, 0.8])

        polygon3 = np.array([[0.7, 0.7, 0.0], [0.9, 0.5, 0.3], [0.5, 0.9, 0.3]])
        polygon3 += np.array([-0.4, -0.3, 0.9])

        #for i, polygon in enumerate([polygon1, polygon2, polygon3]):
        for i, polygon in enumerate([polygon1, polygon2]):
            q_init = np.ones(7) * 0.3
            target_obj_pos = np.ones(3)

            sol = kinsol.solve(q_init, polygon, target_obj_pos)
            assert sol.success 
            assert len(sol.x) == 7 + 3 * use_base

            ineq, eq = kinsol.configuration_constraint_from_polygon(polygon)
            P, jac = kinsol.forward_kinematics(sol.x)
            pos = P[:, :3]
            ineq, eq = polygon_to_trans_constraint(polygon)
            assert ineq.is_satisfying(pos.flatten())
            assert eq.is_satisfying(pos.flatten())
