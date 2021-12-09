#!/usr/bin/env python
import argparse
import numpy as np
from yamaopt.solver import KinematicSolver, SolverConfig
from yamaopt.polygon_constraint import polygon_to_trans_constraint
from visualizer import VisManager

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-robot', type=str, default='fetch', help='robot name')
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

    polygon = np.array([[0.5, -0.6, 0.0], [0.5, 0.6, 0.0], [0.0, 0.0, 0.9]])
    polygon += np.array([0.3, -0.7, 0.7])
    target_pos = np.array([-0.3, -0.6, 1.6])

    q_init = -np.ones(len(kinsol.control_joint_ids)) * 0.4
    sol = kinsol.solve(q_init, polygon, target_pos)
    assert sol.success

    if visualize:
        # visualize
        vm = VisManager(config)
        vm.add_polygon(polygon)
        vm.add_target(target_pos)
        vm.set_angle_vector(sol.x)
        vm.show_while()
