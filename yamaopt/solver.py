import os
from tinyfk import RobotModel
import skrobot
from skrobot.planner.utils import scipinize
from skrobot.planner.utils import _forward_kinematics
from geometry_msgs.msg import PolygonStamped, Polygon, Point32
import yaml
import numpy as np
import scipy.optimize

from yamaopt.polygon_constraint import polygon_to_constraint

class KinematicSolver:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        urdf_path = os.path.expanduser(config['urdf_path'])
        self.kin = RobotModel(urdf_path)

        robot = skrobot.model.RobotModel()
        robot.load_urdf_file(urdf_path)
        self.robot = robot

        # create joint-id, link-id tables
        all_joint_names = [j.name for j in self.robot.joint_list]
        all_link_names = [l.name for l in self.robot.link_list]
        tinyfk_joint_ids = self.kin.get_joint_ids(all_joint_names)
        tinyfk_link_ids = self.kin.get_link_ids(all_link_names)

        self.joint_id_table = {n: id for (n, id) in zip(all_joint_names, tinyfk_joint_ids)}
        self.link_id_table = {n: id for (n, id) in zip(all_link_names, tinyfk_link_ids)}

        self.control_joint_ids = [self.joint_id_table[name] for name in config['control_joint_names']]
        self.end_effector_id = self.link_id_table[config['endeffector_link_name']]

    # TODO lru cache
    def forward_kinematics(self, q):
        assert isinstance(q, np.ndarray) and q.ndim == 1
        with_jacobian = True 
        use_rotation = False # TODO add rotation
        use_base = False
        
        link_ids = [self.end_effector_id]
        joint_ids = self.control_joint_ids
        P, J = self.kin.solve_forward_kinematics(
                [q], link_ids, joint_ids, use_rotation, use_base, with_jacobian)
        return P, J

    def create_objective_function(self, target_obs_pos):

        def f(q):
            P, J = self.forward_kinematics(q)
            val = np.sum((P.flatten() - target_obs_pos) ** 2)
            grad = 2 * (P.flatten() - target_obs_pos).dot(J)
            return val, grad

        return f

    def configuration_constraint_from_polygon(self, np_polygon):
        lin_ineq, lin_eq = polygon_to_constraint(np_polygon)

        def ineq_constraint(q):
            P, J = self.forward_kinematics(q)
            val = ((lin_ineq.A.dot(P.T)).T - lin_ineq.b).flatten()
            jac = lin_ineq.A.dot(J)
            return val, jac

        def eq_constraint(q):
            P, J = self.forward_kinematics(q)
            val = ((lin_eq.A.dot(P.T)).T - lin_eq.b).flatten()
            jac = lin_eq.A.dot(J)
            return val, jac

        return ineq_constraint, eq_constraint

    def solve(self, q_init, np_polygon, target_obs_pos):
        f_ineq, f_eq = self.configuration_constraint_from_polygon(np_polygon)

        eq_const_scipy, eq_const_jac_scipy = scipinize(f_eq)
        eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
                   'jac': eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = scipinize(f_ineq)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                     'jac': ineq_const_jac_scipy}

        f_obj = self.create_objective_function(target_obs_pos)

        f, jac = scipinize(f_obj)

        res = scipy.optimize.minimize(
            f, q_init, method='SLSQP', jac=jac,
            constraints=[eq_dict, ineq_dict])
            #options=slsqp_option)
            #bounds=bounds,
        return res
