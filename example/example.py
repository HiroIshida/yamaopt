#!/usr/bin/env python
import argparse
import os
import time
import trimesh
import skrobot
from skrobot.model.primitives import MeshLink, Sphere
from skrobot.planner.utils import set_robot_config
import numpy as np
from yamaopt.solver import KinematicSolver, SolverConfig
from yamaopt.polygon_constraint import polygon_to_trans_constraint

class VisManager:
    def __init__(self, config):
        urdf_path = os.path.expanduser(config.urdf_path)
        robot = skrobot.model.RobotModel()
        robot.load_urdf_file(urdf_path)

        self.config = config
        self.robot = robot
        self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
        self.viewer.add(robot)

    def add_polygon(self, np_polygon):
        mesh = visual_mesh=trimesh.Trimesh(
                vertices=np_polygon, faces=[[0, 1, 2]], face_colors=[255, 0, 0, 200])
        polygon_link = MeshLink(mesh)
        self.viewer.add(polygon_link)

    def add_target(self, target_pos):
        target_sphere_link = Sphere(0.05, pos=target_pos, color=[0, 0, 255])
        self.viewer.add(target_sphere_link)

    def set_angle_vector(self, q):
        assert len(q) == len(self.config.control_joint_names)
        joints = [self.robot.__dict__[name] for name in self.config.control_joint_names]
        set_robot_config(self.robot, joints, q, with_base=False)

    def show_while(self):
        self.viewer.show()

        print('==> Press [q] to close window')
        while not self.viewer.has_exit:
            time.sleep(0.1)
            self.viewer.redraw()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-robot', type=str, default='fetch', help='robot name')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    args = parser.parse_args()
    robot_name = args.robot
    visualize = args.visualize

    if robot_name == 'fetch':
        config_path = "../config/fetch_conf.yaml"
    elif robot_name == 'pr2':
        config_path = "../config/pr2_conf.yaml"
    else:
        raise Exception()


    config = SolverConfig.from_config_path(config_path)
    kinsol = KinematicSolver(config)

    polygon = np.array([[0.5, -0.6, 0.0], [0.5, 0.6, 0.0], [0.0, 0.0, 0.9]])
    polygon += np.array([0.3, -0.7, 0.7])
    target_pos = np.array([-0.3, -0.6, 1.6])

    sol = kinsol.solve(-np.ones(kinsol.dof)*0.4, polygon, target_pos)
    assert sol.success

    if visualize:
        # visualize
        vm = VisManager(config)
        vm.add_polygon(polygon)
        vm.add_target(target_pos)
        vm.set_angle_vector(sol.x)
        vm.show_while()
