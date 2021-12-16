#!/usr/bin/env python
import argparse
import math
import numpy as np
from skrobot.coordinates.math import rotation_matrix
from yamaopt.solver import KinematicSolver, SolverConfig
from yamaopt.polygon_constraint import polygon_to_trans_constraint
from visualizer import VisManager

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-robot', type=str, default='pr2', help='robot name')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    parser.add_argument('--use_base', action='store_true', help='with base')
    args = parser.parse_args()
    robot_name = args.robot
    visualize = args.visualize
    use_base = args.use_base

    if robot_name == 'fetch':
        config_path = "../config/fetch_conf.yaml"
    elif robot_name == 'pr2':
        config_path = "../config/pr2_conf.yaml"
    else:
        raise Exception()


    config = SolverConfig.from_config_path(config_path, use_base=use_base)
    kinsol = KinematicSolver(config)

    polygon1 = np.array([[1.0, -0.5, -0.5], [1.0, 0.5, -0.5], [1.0, 0.5, 0.5], [1.0, -0.5, 0.5]]) + np.array([0, 0, 1.0])
    polygon2 = polygon1.dot(rotation_matrix(math.pi / 2.0, [0, 0, 1.0]).T)
    polygon3 = polygon1.dot(rotation_matrix(-math.pi / 2.0, [0, 0, 1.0]).T)
    polygons = [polygon1, polygon2, polygon3]
    target_obj_pos = np.array([-0.1, 0.7, 0.3])

    q_init = -np.ones(len(kinsol.control_joint_ids)) * 0.4
    sol, target_polygon = kinsol.solve_multiple(q_init, polygons, target_obj_pos)
    assert sol.success

    if visualize:
        # visualize
        vm = VisManager(config)
        vm.add_polygon_list(polygons)
        vm.add_target(target_obj_pos)
        vm.set_angle_vector(sol.x)
        vm.show_while()
