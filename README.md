# face-lock project
Computer Vision, OpenCV Task

This task is a computer vision class task, a project using opencv only.
This unlocking system uses classical computer vision techniques for face and hand gesture recognition.

When you use OpenCV to recognize your face and recognize user-defined finger gestures, Mosaic processing will be turned off and your face will be exposed.



## Key Features
- Face recognition (LBPH algorithm)
- Facial Mosaic Processing (Locked)
- Hand gesture recognition (based on the number of fingers)
- Unlock with face + hand gesture combination
- Real-time webcam frame processing



## technology of use
- Python 3
- OpenCV
- NumPy
- Haar Cascade (Face Detection)
- LBPH Face Recognizer



## How it works

### 2. Face Learning (train.py )

- Prepare folders named `User_01`, `User_02`, etc., with face images.
- Face detection and learning
- The trained model is saved as 'face_model.yml'.



### 3. Real-Time Execution (main.py)
Run the script to start real-time face and hand gesture recognition using your webcam:
python main.py

When a face is detected, the person's label will be displayed.

If a hand is detected near the face (to the left side), the system will count the number of fingers.

When the correct finger gesture sequence (e.g., 0 → 5 → 2) is recognized, the system removes the mosaic from the detected face, effectively unlocking it.

### Gesture Authentication Logic
gesture_sequence = [0, 5, 2]  # Required finger gestures in order

frame_count_for_gesture = 3   # Number of frames each gesture must be held

gesture_delay = 0.5           # Minimum delay between gestures (in seconds)

Only gestures made on the left side of the detected face are considered.

If the correct sequence is performed consistently, the system unlocks that face.



## Test Environment
Compatible with both macOS and Windows

Python 3.8+

OpenCV version 4.5 or higher



## Usage
### Installation
    conda create -n face-lock python=3.8 -y
    conda activate face-lock
    pip install -r requirements.txt
    pip install opencv-python numpy
    mkdir user_01
    mkdir user_02
    Place the 'haarcascade_frontalface_default.xml' file in the project directory.
    python train.py
    python main.py



### Data Preparation
    face-lock/
    ├── haarcascade_frontalface_default.xml 
    ├── face_model.yml
    ├── main.py
    ├── train.py
    ├── README.md
    ├── User_01/ # Learning Image Folder
    │ ├── 1.jpg
    │ ├── 2.jpg
    │ └── ...
    ├── User_02/
    │ ├── 1.jpg
    │ ├── 2.jpg
    │ └── ...



## Tip
- For improved hand gesture recognition accuracy, test in a well-lit environment with a simple background.
- Keep your hand positioned to the left side of your face.
- The gesture sequence [0 → 5 → 2] can be modified if needed (edit it in the main.py file).
- Hand gesture recognition is performed using contour analysis and convex hulls to estimate the number of fingers.
- Before running train.py, ensure that you have placed sufficient face images into the User_XX/ folders for accurate training.
