#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher = self.create_publisher(Image, '/video_frames', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('/ros_ws/demo.mp4')
        
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open video file")
            raise RuntimeError("Video file not found")
            
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timer = self.create_timer(1.0/fps, self.publish_frame)
        self.get_logger().info(f"Publishing video at {fps} FPS")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher.publish(msg)
        else:
            self.get_logger().info("Video ended")
            self.timer.cancel()
            self.cap.release()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    publisher = VideoPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()