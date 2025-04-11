FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git wget python3-pip vim
RUN pip3 install setuptools==58.2.0
RUN sudo pip3 install tornado
RUN sudo pip3 install simplejpeg  # recommended, but ROSboard can fall back to cv2 or PIL instead

COPY . /ros_ws

RUN cd /ros_ws/src && \
    source /opt/ros/humble/setup.bash && \
	rosdep install -i --from-path ./ --rosdistro humble -y && \
    colcon build && \
	source install/setup.bash 
RUN chmod +x /ros_ws/ros_entrypoint.sh
RUN chmod +x /ros_ws/src/run

ENTRYPOINT ["/ros_ws/ros_entrypoint.sh"]
CMD ["bash"]