
# 🛑 Drowsiness Detection System

A real-time computer vision-based system to detect drowsiness using Eye Aspect Ratio (EAR), helping prevent accidents due to fatigue or microsleep. This project is ideal for use in transportation safety, workplace monitoring, and personal well-being.

## 📌 Features

- Real-time face and eye detection using a webcam
- Eye Aspect Ratio (EAR) calculation using dlib’s 68-point facial landmark detection
- Audible alarm and on-screen visual alerts upon detecting drowsiness
- Lightweight and easy to deploy
- Written in Python with minimal dependencies

## 🧰 Technologies Used

- Python
- OpenCV
- dlib
- SciPy
- playsound

## 🧠 How It Works

1. The webcam captures real-time video.
2. The system detects the face and extracts eye landmarks using `dlib`.
3. The Eye Aspect Ratio (EAR) is calculated using key points around the eyes.
4. If EAR remains below a defined threshold for a set number of frames, drowsiness is detected.
5. An alarm is triggered using `playsound`, along with optional visual warnings.

## 📸 Eye Aspect Ratio (EAR) Formula

\[
EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 \times ||p1 - p4||}
\]

Where \(p1\) to \(p6\) are specific eye landmarks.

## 📂 Project Structure

├── drowsiness\_detector.py
├── shape\_predictor\_68\_face\_landmarks.dat
├── alarm.wav
├── requirements.txt
└── README.md

## ▶️ Getting Started

### Prerequisites

- Python 3.x
- dlib and cmake installed on your system

### Installation

1. Clone the repository:
   git clone https://github.com/your-username/drowsiness-detector.git
   cd drowsiness-detector

2. Install required packages:

   pip install -r requirements.txt

3. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's official source](http://dlib.net/files/) and place it in the project folder.

### Run the Project

python drowsiness_detector.py

## ⚙️ Customization

* You can adjust the EAR threshold and frame count in the script for sensitivity.
* Replace `alarm.wav` with your preferred alert sound.


## 👨‍💻 Author

*Vivek Kumar Chaurasiya*
📧 [vkch849@gmail.com](mailto:vkch849@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/vivek-kumar-chaurasiya-944187220/)

## 📝 License

This project is licensed under the MIT License.
