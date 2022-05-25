import os
import pybullet as pb
import pybullet_data
from moviepy.editor import ImageSequenceClip


class PybulletVisualizer:
    def __init__(self, urdf_path, control_joint_names, use_gui=False):
        client = pb.connect(pb.GUI if use_gui else pb.DIRECT)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        robot_id = pb.loadURDF(urdf_path)
        pbdata_path = pybullet_data.getDataPath()
        #pb.loadURDF(os.path.join(pbdata_path, "samurai.urdf"))

        link_table = {pb.getBodyInfo(robot_id, physicsClientId=client)[0].decode('UTF-8'): -1}
        joint_table = {}
        def heck(path): return "_".join(path.split("/"))
        for _id in range(pb.getNumJoints(robot_id, physicsClientId=client)):
            joint_info = pb.getJointInfo(robot_id, _id, physicsClientId=client)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode('UTF-8')
            joint_table[joint_name] = joint_id
            name_ = joint_info[12].decode('UTF-8')
            name = heck(name_)
            link_table[name] = _id

        self.client = client
        self.robot_id = robot_id
        self.control_joint_ids = [joint_table[name] for name in control_joint_names]

    def set_joint_angle(self, q):
        assert len(self.control_joint_ids) == len(q)

        for angle, joint_id in zip(q, self.control_joint_ids):
            pb.resetJointState(self.robot_id, joint_id, angle,
                               targetVelocity=0.0,
                               physicsClientId=self.client)

    def visualize_sequence(self, q_seq):
        img_seq = []
        for q in q_seq:
            self.set_joint_angle(q)
            img, _ = self.take_photo()
            img_seq.append(img)

        print("dumping gif...")
        clip = ImageSequenceClip([img for img in img_seq], fps=5)
        clip.write_gif("hoge.gif", fps=5)

    def take_photo(self, resolution=1024):
        viewMatrix = pb.computeViewMatrix(
            cameraEyePosition=[-0.5, -3.0, 1.0],
            cameraTargetPosition=[0.8, 0, 1.0],
            cameraUpVector=[0, 0, 1])

        projectionMatrix = pb.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.01,
            farVal=5.1)

        width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
            width=resolution,
            height=resolution,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
        return rgbImg, depthImg
