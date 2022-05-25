#!/usr/bin/env python
import argparse

import numpy as np

from yamaopt.solver import KinematicSolver
from yamaopt.solver import SolverConfig
from yamaopt.visualizer import VisManager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-robot', type=str, default='pr2', help='robot name')
    parser.add_argument('-hover', type=float, default='0.05', help='hover distance [m]')
    parser.add_argument('-shrink', type=float, default='0.05', help='polygon shrink distance [m]')
    parser.add_argument('-margin', type=float, default='5.0', help='joint limit margin [deg]')
    parser.add_argument('--visualize', action='store_true', help='visualize')
    parser.add_argument('--use_base', action='store_true', help='with base')
    parser.add_argument('--limit_base', action='store_true', help='limit movable area of base')
    args = parser.parse_args()
    robot_name = args.robot
    visualize = args.visualize
    use_base = args.use_base
    d_hover = args.hover
    polygon_shrink = args.shrink
    joint_limit_margin = args.margin
    limit_base = args.limit_base

    if robot_name == 'fetch':
        config_path = "../config/fetch_conf.yaml"
    elif robot_name == 'pr2':
        config_path = "../config/pr2_conf.yaml"
    else:
        raise Exception()

    config = SolverConfig.from_config_path(config_path, use_base=use_base, optframe_xyz_from_ef=[0, 0.5, 0.])
    kinsol = KinematicSolver(config)

    polygon = np.array([[0.5, -0.6, 0.0], [0.5, 0.6, 0.0], [0.0, 0.0, 0.9]])
    polygon += np.array([0.3, -0.7, 0.7])
    target_pos = np.array([-0.3, -0.6, 1.6])

    if use_base and limit_base:
        movable_polygon = np.array(
            [[0.5, 0, 0], [0, 0.5, 0], [-0.5, 0, 0], [0, -0.5, 0]])
        movable_polygon += np.array([-1.0, -1.0, 0.0])
    else:
        movable_polygon = None

    q_init = -np.ones(len(kinsol.control_joint_ids)) * 0.4
    sol = kinsol.solve(q_init, polygon, target_pos,
                       movable_polygon=movable_polygon,
                       d_hover=d_hover, polygon_shrink=polygon_shrink,
                       joint_limit_margin=joint_limit_margin)
    if sol.success:
        print('optimization succeeded.')
    else:
        print('optimization failed.')
        assert sol.success

    if visualize:
        # visualize
        vm = VisManager(config)
        vm.add_target(target_pos)
        vm.reflect_solver_result(sol, [polygon], movable_polygon,
                                 show_polygon_axis=True)
        vm.show_while()
