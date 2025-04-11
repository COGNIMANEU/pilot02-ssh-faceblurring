# Dockerfile
FROM ros:humble

# Install all required dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-rqt-image-view \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /ros_ws
COPY . /ros_ws/

# Install Python dependencies
RUN pip3 install opencv-python-headless numpy

# Build the ROS2 package (if you have one)
# RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
#     colcon build --symlink-install

CMD ["bash", "-c", "source /opt/ros/humble/setup.sh && python3 /ros_ws/video_publisher.py"]