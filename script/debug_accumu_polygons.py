#!/usr/bin/env python
import rospy
import pickle
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import PolygonStamped

class Accumulator:
    def __init__(self):
        topic_name = "/organized_multi_plane_segmentation/output_polygon"
        self.sub = rospy.Subscriber(topic_name, PolygonArray, self.callback)
        self.accum_polygons = []
        self.dump_filename = "accum_polygons.pickle"

    def callback(self, msg: PolygonArray):
        rospy.loginfo("accumulating {} polygons".format(len(msg.polygons)))
        self.accum_polygons.extend(msg.polygons)

    def dump_pickle(self):
        rospy.loginfo("dumping {0} polygons to {1}".format(len(self.accum_polygons), self.dump_filename))
        with open(self.dump_filename, 'wb') as f:
            pickle.dump(self.accum_polygons, f)

rospy.init_node('listener', anonymous=True, disable_signals=True)
acc = Accumulator()
try:
    rospy.loginfo("start node")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
except KeyboardInterrupt:
    acc.dump_pickle()
