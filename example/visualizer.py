import os
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

    def add_polygon(self, np_polygon):
        mesh = visual_mesh=trimesh.Trimesh(
                vertices=np_polygon, faces=[[0, 1, 2]], face_colors=[255, 0, 0, 200])
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
