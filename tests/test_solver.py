import os
import numpy as np
from numpy.lib.twodim_base import eye
from yamaopt.solver import KinematicSolver

def compute_numerical_jacobian(f, x0):
    f0 = f(x0)

    dim_inp = len(x0)
    dim_out = len(f0)

    eps =1e-6
    one_hots = [vec * eps for vec in np.eye(dim_inp)]
    jac = np.zeros((dim_out, dim_inp))

    for i, dx in enumerate(one_hots):
        jac[:, i] = (f(x0 + dx) - f(x0)) / eps
    return jac

def test_compute_numerical_jacobian():
    f = lambda x: np.array([np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(x ** 2))])
    x0 = np.random.randn(3)
    jac_numel = compute_numerical_jacobian(f, x0)

    jac_real = np.array([x0 / np.sqrt(np.sum(x0 ** 2)), x0 / np.sqrt(np.sum(x0 ** 2))])
    np.testing.assert_almost_equal(jac_real, jac_numel, decimal=4)


def test_constraint():
    config_path = "./config/pr2_conf.yaml"
    kinsol = KinematicSolver(config_path)

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
