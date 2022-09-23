#!/usr/bin/env python
""" Draw robotic grippers using RVIZ markers and make them interactive.
PUBLISHES:
visualization_marker1 (visualization_messages/Marker) - The markers that we are drawing
SUBSCRIBES:
Subscribes to an interactive marker server
"""

import rospy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer, InteractiveMarker 
from visualization_msgs.msg import Marker, InteractiveMarkerControl, MarkerArray


class Gripper(object):
    def __init__(self):
        self.pub1 = rospy.Publisher("visualization_marker1", Marker, queue_size = 2)
        # The second link is oriented at 90 degrees
        self.m = Marker()
        self.m.header.frame_id = "base"
        self.m.header.stamp = rospy.Time()
        self.m.id = 1
        self.m.type = Marker.CYLINDER
        self.m.action = Marker.ADD
        self.m.scale.x = 1.0
        self.m.scale.y = 1.0
        self.m.scale.z = 3.0
        self.m.pose.position.x = 5
        self.m.pose.position.y = 2
        self.m.pose.position.z = -1
        self.m.pose.orientation.x = .707
        self.m.pose.orientation.y = 0
        self.m.pose.orientation.z = 0
        self.m.pose.orientation.w = .707
        self.m.color.r = 0
        self.m.color.g = 1
        self.m.color.b = 1
        self.m.color.a = 1
        self.pub1.publish(self.m)

        self.m1 = Marker()
        self.m1.header.frame_id = "base"
        self.m1.header.stamp = rospy.Time()
        self.m1.id = 2
        self.m1.type = Marker.CYLINDER
        self.m1.action = Marker.ADD
        self.m1.scale.x = 1.0
        self.m1.scale.y = 1.0
        self.m1.scale.z = 3.0
        self.m1.pose.position.x = -5
        self.m1.pose.position.y = 2
        self.m1.pose.position.z = -1
        self.m1.pose.orientation.x = .707
        self.m1.pose.orientation.y = 0
        self.m1.pose.orientation.z = 0
        self.m1.pose.orientation.w = .707
        self.m1.color.r = 1
        self.m1.color.g = 1
        self.m1.color.b = 0
        self.m1.color.a = 1
        self.pub1.publish(self.m1)

        self.server = InteractiveMarkerServer("gripper_marker")

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "base"
        int_marker.name = "gripper"
        int_marker.description = "Move to open/close the gripper"
        int_marker.pose.orientation.w = .707
        int_marker.pose.orientation.z = .707

        box_marker = Marker()
        box_marker.type = Marker.SPHERE
        box_marker.scale.x = 0.5
        box_marker.scale.y = 0.5
        box_marker.scale.z = 0.5
        box_marker.color.r = 1.0
        box_marker.color.g = 0.0
        box_marker.color.b = 0.0
        box_marker.color.a = 1.0

        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)


        int_marker.controls.append(box_control)

        speed_control = InteractiveMarkerControl()
        speed_control.name = "move"
        speed_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS

        int_marker.controls.append(speed_control)

        self.server.insert(int_marker, self.callback)
        self.server.applyChanges()

    def callback(self,feedback):
        """ Callback for interactive markers.  feedback contains the pose of the marker from rviz """
        w = feedback.pose.orientation.w
        self.m.pose.position.x = 3*w
        self.m1.pose.position.x = -self.m.pose.position.x
        self.m.action = Marker.MODIFY
        self.m1.action = Marker.MODIFY
        self.pub1.publish(self.m)
        self.pub1.publish(self.m1)

if __name__ == "__main__":
    rospy.init_node("gripper")   

    g = Gripper()

    rospy.spin()