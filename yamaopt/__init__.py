import os
import shutil
import xml.etree.ElementTree as ET
from skrobot.data import pr2_urdfpath, fetch_urdfpath

# bit dirty, but we will probably use only pr2 and fetch, so...

def add_sensor_frame_to_urdf(robot_urdf_path, sensor_urdf_path, out_urdf_path):
    # robot_root, sensor_root: ElementTree object
    # robot, links, joints: Element object <robot>, <link>, <joint>
    robot_root = ET.parse(robot_urdf_path)
    robot = robot_root.getroot()
    sensor_root = ET.parse(sensor_urdf_path)
    links = sensor_root.findall('link')
    joints = sensor_root.findall('joint')
    for link in links:
        robot.append(link)
    for joint in joints:
        robot.append(joint)
    robot_root.write(out_urdf_path)

# Create urdf under ~/.skrobot
for robot in ['pr2', 'fetch']:
    urdf_path = os.path.expanduser('~/.skrobot/{}_description'.format(robot))
    if not os.path.exists(urdf_path):
        # Create {robot}.urdf
        print("downloading pr2 model... This takes place only once.")
        if robot == 'pr2':
            pr2_urdfpath()
        elif robot == 'fetch':
            fetch_urdfpath()
        # Create {robot}_sensors.urdf for test
        data_dir = os.path.abspath(
            os.path.dirname(__file__) + '/../tests/data')
        robot_urdf_path = urdf_path + '/{}.urdf'.format(robot)
        sensor_urdf_path = data_dir + '/{}_sensor_frames.urdf'.format(robot)
        out_urdf_path = urdf_path + '/{}_sensors.urdf'.format(robot)
        print('Create {}'.format(out_urdf_path))
        add_sensor_frame_to_urdf(
            robot_urdf_path, sensor_urdf_path, out_urdf_path)
