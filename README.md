# Pilot02 SSH Face Blurring

This component provides a modular and configurable face anonymization system for use in ROS2 environments. It ensures privacy by detecting and blurring faces in video streams using various supported backends. Designed for flexibility, it enables switching between different face detection algorithms and anonymization styles based on the performance and accuracy needs of the deployment.

## Features

- Modular face detection and anonymization pipeline.
- ROS2 integration for processing image topics in real time.
- Supports video files, live streams, and raw image frames.
- Dockerized setup for easy deployment and testing.
- Graceful fallback to original frames on failure.

## Supported Blurring Methods

- **Haar Cascades (OpenCV)**  
  Classical real-time face detection suitable for low-resource systems.

- **YOLOv5 (Ultralytics)**  
  Deep-learning detector offering high accuracy and performance.  
  [GitHub →](https://github.com/ultralytics/yolov5)

- **MTCNN**  
  A multi-task cascaded convolutional network optimized for face detection.  
  [GitHub →](https://github.com/ipazc/mtcnn)

- **deface**  
  Command-line anonymization tool supporting multiple filter modes (blur, mosaic, solid).  
  [GitHub →](https://github.com/ORB-HD/deface)


## Guidelines for build and test the component 

### 1. **Build the Main Docker Image:**

In this step, we build the Docker image using the provided `Dockerfile`. The image is named `pilot02-ssh-faceblurring`.

```bash
cd faceblurring
docker build -t pilot02-ssh-faceblurring .
```

### 2. **Build and Run the test automation:**

Test automation is integrated by docker-compose file:

Run: 
```bash
docker-compose up --build
```

After execution, you will be able to see at [Rosboard →](https://localhost:8888) this demo (download video at https://raw.githubusercontent.com/COGNIMANEU/pilot02-ssh-faceblurring/main/test/faceblurringtest.mp4):

![Face Blurring Demo](./test/faceblurringtest.gif)

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## License

This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE. Depending on the usage, additional licenses must be considered:

- BSD 3-Clause License (when using rosboard)
- MIT License (when using deface)
- Intel License Agreement For Open Source Computer Vision Library (when using haar)
- MIT License (when using mtcnn)
- Same GNU AFFERO GENERAL PUBLIC LICENSE (when using yolov5)

See the [LICENSE](LICENSE) file for details.