import os
from tinyfk import RobotModel
import skrobot
from skrobot.planner.utils import scipinize
from skrobot.planner.utils import _forward_kinematics
from geometry_msgs.msg import PolygonStamped, Polygon, Point32
import yaml
import numpy as np
import scipy.optimize

from yamaopt.polygon_constraint import polygon_to_trans_constraint
from yamaopt.polygon_constraint import polygon_to_desired_rpy
from yamaopt.visualize import PybulletVisualizer

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

        self.config = config
        self.control_joint_ids = [self.joint_id_table[name] for name in config['control_joint_names']]
        self.end_effector_id = self.link_id_table[config['endeffector_link_name']]

    # TODO lru cache
    def forward_kinematics(self, q):
        assert isinstance(q, np.ndarray) and q.ndim == 1
        with_jacobian = True 
        use_rotation = True
        use_base = False
        
        link_ids = [self.end_effector_id]
        joint_ids = self.control_joint_ids
        P, J = self.kin.solve_forward_kinematics(
                [q], link_ids, joint_ids, use_rotation, use_base, with_jacobian)
        return P, J

    def create_objective_function(self, target_obs_pos):

        def f(q):
            P_whole, J_whole = self.forward_kinematics(q)
            P_pos = P_whole[:, :3]
            J_pos = J_whole[:3, :]
            val = np.sum((P_pos.flatten() - target_obs_pos) ** 2)
            grad = 2 * (P_pos.flatten() - target_obs_pos).dot(J_pos)
            return val, grad

        return f

    def configuration_constraint_from_polygon(self, np_polygon):
        lin_ineq, lin_eq = polygon_to_trans_constraint(np_polygon)
        rpy_desired = polygon_to_desired_rpy(np_polygon)
        print("desired")
        print(rpy_desired)

        def ineq_constraint(q):
            P_whole, J_whole = self.forward_kinematics(q)
            P_pos = P_whole[:, :3]
            J_pos = J_whole[:3, :]
            val = ((lin_ineq.A.dot(P_pos.T)).T - lin_ineq.b).flatten()
            jac = lin_ineq.A.dot(J_pos)
            return val, jac

        def eq_constraint(q):
            P_whole, J_whole = self.forward_kinematics(q)
            P_pos, P_rot = P_whole[:, :3], P_whole[:, 3:]
            J_pos, J_rot = J_whole[:3, :], J_whole[3:, :]
            val_pos = ((lin_eq.A.dot(P_pos.T)).T - lin_eq.b).flatten()
            jac_pos = lin_eq.A.dot(J_pos)

            val_rot = P_rot.flatten() - rpy_desired
            jac_rot = J_rot
            return np.hstack([val_pos, val_rot]), np.vstack([jac_pos, jac_rot])

        return ineq_constraint, eq_constraint

    def solve(self, q_init, np_polygon, target_obs_pos, output_gif=False):
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

        if output_gif:
            urdf_path = os.path.expanduser(self.config['urdf_path'])
            vis = PybulletVisualizer(urdf_path, self.config['control_joint_names'], False)
            n_seq = 20
            dq = (res.x - q_init) / (n_seq - 1)
            q_seq = [q_init + dq * i for i in range(n_seq)]
            vis.visualize_sequence(q_seq)

        return res
