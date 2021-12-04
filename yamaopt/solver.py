import os
from tinyfk import RobotModel
import skrobot
from skrobot.planner.utils import _forward_kinematics
from geometry_msgs.msg import PolygonStamped, Polygon, Point32
import yaml
import numpy as np

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

    def configuration_constraint_from_polygon(self, np_polygon, use_base=False):
        lin_ineq, lin_eq = polygon_to_constraint(np_polygon)

        # TODO lru cache
        def fk(q):
            assert isinstance(q, np.ndarray) and q.ndim == 1
            with_jacobian = True 
            use_rotation = False # TODO add rotation
            
            link_ids = [self.end_effector_id]
            joint_ids = self.control_joint_ids
            P, J = self.kin.solve_forward_kinematics(
                    [q], link_ids, joint_ids, use_rotation, use_base, with_jacobian)
            return P, J

        def create_ineq_constraint(q):
            P, J = fk(q)
            val = ((lin_ineq.A.dot(P.T)).T - lin_ineq.b).flatten()
            jac = lin_ineq.A.dot(J)
            return val, jac

        def create_eq_constraint(q):
            P, J = fk(q)
            val = ((lin_eq.A.dot(P.T)).T - lin_eq.b).flatten()
            jac = lin_eq.A.dot(J)
            return val, jac

        return create_ineq_constraint, create_eq_constraint

