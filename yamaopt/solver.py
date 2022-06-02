import copy
import math
import os

import attr
import numpy as np
import scipy.optimize
import skrobot
from skrobot.model.joint import LinearJoint
from skrobot.model.joint import RotationalJoint
import sympy
from tinyfk import RobotModel
import yaml

from yamaopt.polygon_constraint import ConcavePolygonException
from yamaopt.polygon_constraint import polygon_to_desired_axis
from yamaopt.polygon_constraint import polygon_to_trans_constraint
from yamaopt.polygon_constraint import ZValueNotZeroException
from yamaopt.pycompat import lru_cache
from yamaopt.utils import expr_axis
from yamaopt.utils import expr_axis_jacobian
from yamaopt.utils import scipinize


@attr.s  # like a dataclass in python3
class SolverConfig(object):
    use_base = attr.ib()
    urdf_path = attr.ib()
    optimization_frame = attr.ib()
    control_joint_names = attr.ib()
    endeffector_link_name = attr.ib()
    optframe_xyz_from_ef = attr.ib()  # ef means end effector
    optframe_rpy_from_ef = attr.ib()

    @classmethod
    def from_config_path(cls,
                         config_path,
                         use_base=False,
                         optframe_xyz_from_ef=None,
                         optframe_rpy_from_ef=None,
                         ):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        if optframe_xyz_from_ef is None:
            optframe_xyz_from_ef = cfg['optframe_xyz_from_ef']
        if optframe_rpy_from_ef is None:
            optframe_rpy_from_ef = cfg['optframe_rpy_from_ef']

        return cls(
            use_base,
            urdf_path=cfg['urdf_path'],
            optimization_frame=cfg['optimization_frame'],
            control_joint_names=cfg['control_joint_names'],
            endeffector_link_name=cfg['endeffector_link_name'],
            optframe_xyz_from_ef=optframe_xyz_from_ef,
            optframe_rpy_from_ef=optframe_rpy_from_ef,
        )


@attr.s  # like a dataclass in python3
class SolverResult(object):
    success = attr.ib(default=None)
    x = attr.ib(default=None)
    fun = attr.ib(default=None)

    # additional infos
    end_coords = attr.ib(default=None)
    optframe_coords = attr.ib(default=None)
    target_polygon = attr.ib(default=None)
    _d_hover = attr.ib(default=None)
    _sol_scipy = attr.ib(default=None)


class KinematicSolver:
    def __init__(self, config):
        urdf_path = os.path.expanduser(config.urdf_path)
        self.kin = RobotModel(urdf_path)

        # Here this model is used only for obtaining joint type (as an urdf parser)
        robot_model = skrobot.model.RobotModel()
        robot_model.load_urdf_file(urdf_path)

        self.config = config
        self.control_joint_ids = self.kin.get_joint_ids(config.control_joint_names)
        joint_limits = self.kin.get_joint_limits(self.control_joint_ids)
        joint_types = [type(robot_model.__dict__[jn]) for jn in config.control_joint_names]

        if self.config.use_base:
            joint_limits.extend([[None, None]] * 3)  # for x, y, theta
            joint_types.extend([LinearJoint, LinearJoint, RotationalJoint])

        self.joint_limits = joint_limits
        self.joint_types = joint_types

        self.endeffector_id = self.kin.get_link_ids([config.endeffector_link_name])[0]
        optimization_frame_name = 'optframe'
        self.kin.add_new_link(optimization_frame_name, self.endeffector_id,
                              config.optframe_xyz_from_ef, config.optframe_rpy_from_ef)
        self.optframe_id = self.kin.get_link_ids([optimization_frame_name])[0]

    @property
    def dof(self):
        return len(self.control_joint_ids) + 3 * (self.config.use_base)

    def forward_kinematics(self, q, link_id=None):
        if link_id is None:
            link_id = self.endeffector_id
        assert isinstance(q, np.ndarray) and q.ndim == 1
        with_jacobian = True
        use_rotation = True
        use_base = self.config.use_base

        link_ids = [link_id]
        joint_ids = self.control_joint_ids
        P, J = self.kin.solve_forward_kinematics(
            [q], link_ids, joint_ids, use_rotation, use_base, with_jacobian)
        return P, J

    def create_objective_function(self, target_obs_pos):

        def f(q):
            P_whole, J_whole = self.forward_kinematics(q, self.optframe_id)
            P_pos = P_whole[:, :3]
            J_pos = J_whole[:3, :]
            val = np.sum((P_pos.flatten() - target_obs_pos) ** 2)
            grad = 2 * (P_pos.flatten() - target_obs_pos).dot(J_pos)
            return val, grad

        return f

    def configuration_constraint_from_polygon(
            self,
            np_polygon,
            normal=None,
            align_axis_name="x",
            d_hover=0.0,
            polygon_shrink=0.0):

        axis_names = ["x", "y", "z"]
        assert align_axis_name in axis_names

        # for pos constraint
        lin_ineq, lin_eq = polygon_to_trans_constraint(
            np_polygon, normal=normal,
            d_hover=d_hover, polygon_shrink=polygon_shrink)

        hand_match_axis_idx = axis_names.index(align_axis_name)
        f_axis = self.get_axis_lambda(hand_match_axis_idx)
        f_axis_jacobian = self.get_axis_jacobian_lambda(hand_match_axis_idx)

        polygon_normal_axis_idx = 0
        axis_desired = polygon_to_desired_axis(np_polygon, polygon_normal_axis_idx, normal=normal)

        def hand_ineq_constraint(q):
            P_whole, J_whole = self.forward_kinematics(q)
            P_pos = P_whole[:, :3]
            J_pos = J_whole[:3, :]
            val = ((lin_ineq.A.dot(P_pos.T)).T - lin_ineq.b).flatten()
            jac = lin_ineq.A.dot(J_pos)
            return val, jac

        def hand_eq_constraint(q):
            # pos value and pos jacobian
            P_whole, J_whole = self.forward_kinematics(q)
            P_pos, P_rot = P_whole[:, :3], P_whole[:, 3:]
            J_pos, J_rot = J_whole[:3, :], J_whole[3:, :]
            val_pos = ((lin_eq.A.dot(P_pos.T)).T - lin_eq.b).flatten()
            jac_pos = lin_eq.A.dot(J_pos)

            rpy = P_rot.flatten()
            axis = f_axis(*rpy).flatten()
            axis_indices_extract = [1, 2]  # without extraction, jacobian will be degenerated

            jac_axis = f_axis_jacobian(*rpy)[axis_indices_extract, :]

            val_rot = axis[axis_indices_extract] - axis_desired[axis_indices_extract]
            jac_rot = jac_axis.dot(J_rot)
            np.hstack([val_pos, val_rot])
            np.vstack([jac_pos, jac_rot])

            return np.hstack([val_pos, val_rot]), np.vstack([jac_pos, jac_rot])

        return hand_ineq_constraint, hand_eq_constraint

    def base_constraint_from_polygon(self, movable_polygon):
        z_values = np.array([m[2] for m in movable_polygon])
        if not np.all(z_values == 0):
            raise ZValueNotZeroException

        b_lin_ineq = polygon_to_trans_constraint(
            movable_polygon, normal=None, d_hover=0.0)[0]

        def base_ineq_constraint(q):
            P = np.array([q[-3:]])
            P[0][-1] = 0  # P = [x, y, theta=0]
            J = np.zeros((3, len(q)))
            J[0][-3] = 1.0
            J[1][-2] = 1.0
            val = ((b_lin_ineq.A.dot(P.T)).T - b_lin_ineq.b).flatten()
            jac = b_lin_ineq.A.dot(J)
            return val, jac

        return base_ineq_constraint

    def solve(
            self,
            q_init,
            np_polygon,
            target_obs_pos,
            movable_polygon=None,
            normal=None,
            align_axis_name="z",
            d_hover=0.0,
            polygon_shrink=0.0,
            joint_limit_margin=0.0):

        if self.config.use_base:
            q_init = np.hstack((q_init, np.zeros(3)))
        else:
            print('WARNING: movable_polygon is given though use_base is False.')
            movable_polygon = None  # Ignore movable area if use_base is False
        assert len(q_init) == self.dof

        try:
            # Constraint functions for hand
            f_ineq, f_eq = self.configuration_constraint_from_polygon(
                np_polygon, normal=normal,
                align_axis_name=align_axis_name,
                d_hover=d_hover, polygon_shrink=polygon_shrink)
            eq_const_scipy, eq_const_jac_scipy = scipinize(f_eq)
            eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
                       'jac': eq_const_jac_scipy}
            ineq_const_scipy, ineq_const_jac_scipy = scipinize(f_ineq)
            ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                         'jac': ineq_const_jac_scipy}
            # Constraint functions for base
            if movable_polygon is None:
                cons = [eq_dict, ineq_dict]
            else:
                b_ineq = self.base_constraint_from_polygon(movable_polygon)
                b_ineq_const_scipy, b_ineq_const_jac_scipy = scipinize(b_ineq)
                b_ineq_dict = {'type': 'ineq', 'fun': b_ineq_const_scipy,
                               'jac': b_ineq_const_jac_scipy}
                cons = [eq_dict, ineq_dict, b_ineq_dict]
        except ConcavePolygonException:
            print("Input polygon is not convex. Skip optimization.")
            return self._create_solver_result_from_scipy_sol(None)
        except ZValueNotZeroException:
            print("movable polygon's z value is not 0. Skip optimization.")
            return self._create_solver_result_from_scipy_sol(None)

        # Objective function
        f_obj = self.create_objective_function(target_obs_pos)
        f, jac = scipinize(f_obj)

        # Bounds
        joint_limits_tight = copy.deepcopy(self.joint_limits)
        margin = joint_limit_margin * math.pi / 180.0
        for i in range(len(joint_limits_tight)):
            joint_type = self.joint_types[i]
            if joint_type == LinearJoint:
                continue
            is_infinite_rotational_joint = (None in joint_limits_tight[i])
            if not is_infinite_rotational_joint:
                joint_limits_tight[i][0] += margin  # tighten lower bound
                joint_limits_tight[i][1] -= margin  # tighten upper bound

        # Solve optimization
        sqp_option = {'maxiter': 300}
        sol = scipy.optimize.minimize(
            f, q_init, method='SLSQP', jac=jac, bounds=joint_limits_tight,
            constraints=cons, options=sqp_option)

        """
        if output_gif:
            urdf_path = os.path.expanduser(self.config.urdf_path)
            vis = PybulletVisualizer(urdf_path, self.config.control_joint_names, False)
            n_seq = 20
            dq = (res.x - q_init) / (n_seq - 1)
            q_seq = [q_init + dq * i for i in range(n_seq)]
            vis.visualize_sequence(q_seq)
        """
        return self._create_solver_result_from_scipy_sol(sol, np_polygon, d_hover)

    def solve_multiple(self,
                       q_init,
                       np_polygons,
                       target_obs_pos,
                       movable_polygon=None,
                       normals=None,
                       align_axis_name="x",
                       d_hover=0.0,
                       polygon_shrink=0.0,
                       joint_limit_margin=0.0):
        """
        np_polygons: List[np.ndarray]
        """
        min_cost = np.inf
        min_sol = None
        target_polygon = None
        if normals is None:
            normals = [None] * len(np_polygons)
        for np_polygon, normal in zip(np_polygons, normals):
            sol = self.solve(
                q_init, np_polygon, target_obs_pos,
                movable_polygon=movable_polygon, normal=normal,
                align_axis_name=align_axis_name,
                d_hover=d_hover, polygon_shrink=polygon_shrink,
                joint_limit_margin=joint_limit_margin)
            if sol.success and sol.fun < min_cost:
                min_cost = sol.fun
                min_sol = sol
                target_polygon = np_polygon
        result = self._create_solver_result_from_scipy_sol(min_sol, target_polygon, d_hover)
        return result

    def _create_solver_result_from_scipy_sol(
            self, sol_scipy, target_polygon=None, d_hover=None):
        if sol_scipy is None:
            return SolverResult(success=False)
        endeffector_pose = self.forward_kinematics(sol_scipy.x)[0][0]
        optframe_pose = self.forward_kinematics(sol_scipy.x, self.optframe_id)[0][0]
        return SolverResult(sol_scipy.success, sol_scipy.x, sol_scipy.fun,
                            endeffector_pose, optframe_pose, target_polygon, d_hover, sol_scipy)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_axis_lambda(axis_idx):
        arg_exprs = sympy.symbols(("roll", "pitch", "yaw"))
        expr = expr_axis(*arg_exprs, axis_idx=axis_idx)
        f_axis = sympy.lambdify(arg_exprs, expr, modules="numpy")
        return f_axis

    @staticmethod
    @lru_cache(maxsize=None)
    def get_axis_jacobian_lambda(axis_idx):
        arg_exprs = sympy.symbols(("roll", "pitch", "yaw"))
        expr = expr_axis_jacobian(*arg_exprs, axis_idx=axis_idx)
        f_jac_axis = sympy.lambdify(arg_exprs, expr, modules="numpy")
        return f_jac_axis
