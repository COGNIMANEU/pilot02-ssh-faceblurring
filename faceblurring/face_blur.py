import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
from mtcnn import MTCNN
import tempfile
import subprocess
import os

class FaceBlurNode(Node):
    def __init__(self, model, input_topic, output_topic):
        super().__init__('face_blur_node')

        self.model = model
        self.input_topic = input_topic
        self.output_topic = output_topic

        # Initialize CvBridge for converting ROS messages to OpenCV images
        self.bridge = CvBridge()

        # Create a publisher for the blurred faces image
        self.blurred_faces_publisher = self.create_publisher(Image, self.output_topic, 10)

        # Create a subscriber to the image topic
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,  # Topic to subscribe to
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image using CvBridge
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Apply face blurring/anonymization based on selected model
        frame = self.process_faces(frame)

        # Publish the resulting image
        self.publish_image(frame)

    def process_faces(self, frame):
        # Use different face processing models based on the input model
        if self.model == "haar":
            frame = self.blur_faces_haar(frame)
        elif self.model == "yolov5":
            frame = self.blur_faces_yolov5(frame)
        elif self.model == "mtcnn":
            frame = self.blur_faces_mtcnn(frame)
        elif self.model == "deface":
            frame = self.blur_faces_deface(frame)
        else:
            self.get_logger().warn(f"Unknown model: {self.model}, defaulting to haar")
            frame = self.blur_faces_haar(frame)
        return frame

    def blur_faces_haar(self, frame):
        # Haar Cascade for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            frame[y:y + h, x:x + w] = blurred_face

        return frame

    def blur_faces_yolov5(self, frame):
        # YOLOv5 for face detection (Assuming YOLO model is loaded correctly)
        # You would load the YOLO model and perform detection here
        # For now, this is a placeholder method
        return frame

    def blur_faces_mtcnn(self, frame):
        # MTCNN for face detection
        detector = MTCNN()
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_face

        return frame

    def blur_faces_deface(self, frame):
        """
        Process frame using deface library for anonymization.
        Saves the image to a temporary file, invokes the deface command for anonymization,
        and returns the anonymized image.
        """
        try:
            # Save the received image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_image_path = tmp_file.name
                cv2.imwrite(temp_image_path, frame)  # Save the image to the temporary file
                
                # Create the path for the anonymized image
                anonymized_image_path = temp_image_path.replace('.jpg', '_anonymized.jpg')
                
                # Run the deface command to anonymize the image
                deface_command = f"deface {temp_image_path} -o {anonymized_image_path}"
                subprocess.run(deface_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Suppress output
                
                # Read the processed (anonymized) image
                anonymized_frame = cv2.imread(anonymized_image_path)
                
                # Remove the temporary files if they are no longer needed
                os.remove(temp_image_path)
                os.remove(anonymized_image_path)
                
                return anonymized_frame
        
        except Exception as e:
            self.get_logger().error(f"Deface processing failed: {str(e)}")
            return frame  # Return the original image if an error occurs

    def publish_image(self, frame):
        # Convert the frame back to a ROS image message
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        # Publish the message
        self.blurred_faces_publisher.publish(img_msg)

def main(args=None):
    """
    Main function for the face blurring ROS 2 node.
    
    Handles argument parsing, node initialization, and ROS 2 lifecycle.
    
    Args:
        args: Arguments passed to rclpy.init(). Not used for CLI arguments.
    """
    # Initialize ROS 2 - this handles ROS-specific arguments (like --ros-args)
    rclpy.init(args=args)
    
    # Get logger instance for this node
    logger = rclpy.logging.get_logger('face_blur_node')
    
    # Log the raw command line arguments for debugging
    logger.info(f"Raw command line args: {sys.argv}")
    
    # Argument parsing logic
    # Note: sys.argv[0] is the script name, actual args start at [1]
    if len(sys.argv) >= 4:
        # We received enough arguments
        model = sys.argv[1]        # 1st arg: model type ('haar', 'yolov5', 'mtcnn', 'deface')
        input_topic = sys.argv[2]  # 2nd arg: input topic name
        output_topic = sys.argv[3] # 3rd arg: output topic name
    else:
        # Default values if not enough arguments provided
        logger.warn("Insufficient arguments, using defaults")
        model = "haar"
        input_topic = "/video_frames"
        output_topic = "/blurred_faces"
    
    # Log the actual configuration being used
    logger.info(
        f"Configuration - Model: {model}, "
        f"Input: {input_topic}, "
        f"Output: {output_topic}"
    )
    
    # Create and run the node
    try:
        node = FaceBlurNode(model, input_topic, output_topic)
        logger.info("Node started, spinning...")
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Node crashed: {str(e)}")
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        logger.info("Node shutdown complete")

if __name__ == '__main__':
    main()