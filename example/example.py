#!/usr/bin/env python
import time
import trimesh
import skrobot
from skrobot.model.primitives import MeshLink, Sphere
import numpy as np
from yamaopt.solver import KinematicSolver
from yamaopt.polygon_constraint import polygon_to_trans_constraint

if __name__=='__main__':
    config_path = "../config/pr2_conf.yaml"
    kinsol = KinematicSolver(config_path)

    polygon = np.array([[0.5, -0.6, 0.0], [0.5, 0.6, 0.0], [0.0, 0.0, 0.9]])
    polygon += np.array([0.3, -0.7, 0.7])

    mesh = visual_mesh=trimesh.Trimesh(vertices=polygon, faces=[[0, 1, 2]], face_colors=[255, 0, 0, 200])
    polygon_link = MeshLink(mesh)

    target_pos = np.array([-0.3, -0.6, 0.6])
    target_sphere_link = Sphere(0.05, pos=target_pos, color=[0, 0, 255])

    sol = kinsol.solve(-np.ones(7)*0.4, polygon, target_pos)
    kinsol.set_skrobot_angle_vector(sol.x)
    assert sol.success

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    viewer.add(kinsol.robot)
    viewer.add(polygon_link)
    viewer.add(target_sphere_link)
    viewer.show()

    print('==> Press [q] to close window')
    while not viewer.has_exit:
        time.sleep(0.1)
        viewer.redraw()
