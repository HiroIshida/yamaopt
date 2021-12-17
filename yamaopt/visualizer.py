import os
import numpy as np
import time
import trimesh
import skrobot
from skrobot.model.primitives import MeshLink, Sphere
from skrobot.planner.utils import set_robot_config

class VisManager:
    def __init__(self, config):
        urdf_path = os.path.expanduser(config.urdf_path)
        robot = skrobot.model.RobotModel()
        robot.load_urdf_file(urdf_path)

        self.config = config
        self.robot = robot
        self.viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
        self.viewer.add(robot)

    def _convert_polygon_to_mesh(self, np_polygon):
        vec0 = np_polygon[1] - np_polygon[0]
        vec1 = np_polygon[2] - np_polygon[1]
        tmp = np.cross(vec0, vec1)
        n_vec = tmp / np.linalg.norm(tmp)
        center = np.mean(np_polygon, axis=0)
        #center_hover = center + n_vec -0.1

        polygon_auged = np.vstack([np_polygon, np_polygon[0]])

        faces = []
        vertices = [center]
        for i in range(len(polygon_auged)-1):
            v0 = polygon_auged[i]
            v1 = polygon_auged[i+1]
            vertices.extend([v0, v1])
            faces.append([0, len(vertices)-2, len(vertices)-1])


        vertices = np.array(vertices)
        return vertices, faces

    def add_polygon(self, np_polygon):
        V, F = self._convert_polygon_to_mesh(np_polygon)
        mesh = visual_mesh=trimesh.Trimesh(
                vertices=V, faces=F, face_colors=[255, 0, 0, 200])
        polygon_link = MeshLink(mesh)
        self.viewer.add(polygon_link)

    def add_polygon_list(self, np_polygon_list):
        for np_polygon in np_polygon_list:
            self.add_polygon(np_polygon)

    def add_target(self, target_pos):
        target_sphere_link = Sphere(0.05, pos=target_pos, color=[0, 0, 255])
        self.viewer.add(target_sphere_link)

    def set_angle_vector(self, q):
        joints = [self.robot.__dict__[name] for name in self.config.control_joint_names]
        set_robot_config(self.robot, joints, q, with_base=self.config.use_base)

    def show_while(self):
        self.viewer.show()

        print('==> Press [q] to close window')
        while not self.viewer.has_exit:
            time.sleep(0.1)
            self.viewer.redraw()
