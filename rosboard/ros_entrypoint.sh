#!/bin/bash
set -e

source /opt/ros/humble/setup.sh
source /ros_ws/src/install/setup.sh

exec "$@"