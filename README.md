# Face and Object Detection with Webcam

This project demonstrates real-time face and object detection using a webcam. It utilizes OpenCV for face detection and TensorFlow Hub's EfficientDet model for object detection.

## Features

- Real-time face detection using Haar cascades.
- Real-time object detection using EfficientDet from TensorFlow Hub.
- Display of bounding boxes and labels around detected faces and objects.

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- TensorFlow Hub

## Installation

1. Clone the repository:

```bash
git clone https://github.com/0day-Yash/face-object-detection.git
cd face-object-detection
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv face-object-detection-env
source face-object-detection-env/bin/activate
```

3. Install the required packages:

```bash
pip install opencv-python tensorflow tensorflow-hub
```

## Usage

1. Ensure your webcam is connected.

2. Run the script:

```bash
python3 deepaiproject.py
```

3. The video feed will display with detected faces and objects. Press `q` to quit the application.

## Code Overview

- `deepaiproject.py`: Main script for capturing video from the webcam and performing face and object detection.

### Face Detection

The face detection is implemented using OpenCV's pre-trained Haar cascades.

### Object Detection

The object detection is implemented using TensorFlow Hub's EfficientDet model.

## Troubleshooting

- If the script fails to access the webcam, ensure it is properly connected and accessible.
- For performance optimization, ensure you have the latest versions of the dependencies.
