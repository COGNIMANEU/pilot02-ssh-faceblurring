FROM ros:humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-venv \
    build-essential \
    libopencv-dev \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    && apt-get clean

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip


# Install Python dependencies (including mtcnn, tensorflow, and deface)
RUN python3 -m pip install --no-cache-dir \
    "numpy<2" \
    "opencv-python" \
    "torch==1.13.1" \
    "mtcnn" \
    "tensorflow==2.10.0" \
    "onnxruntime" \  
    "protobuf<4" \
    "git+https://github.com/ORB-HD/deface.git"

# Create working directory
WORKDIR /app

# Copy the Python script
COPY face_blur.py .

# Default command to run the Python script
CMD ["python3", "face_blur.py"]