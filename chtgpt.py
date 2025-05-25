import cv2
import dlib
import os
from scipy.spatial import distance
import pygame
import time

# Initialize pygame mixer
pygame.mixer.init()

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play(-1)  # Loop indefinitely

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Function to calculate Eye Aspect Ratio
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to draw eye landmarks
def draw_eye(frame, landmarks, start, end):
    eye = []
    for n in range(start, end):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        eye.append((x, y))
        next_point = n + 1 if n != end - 1 else start
        x2 = landmarks.part(next_point).x
        y2 = landmarks.part(next_point).y
        cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
    return eye

# EAR Threshold and Timing
EAR_THRESHOLD = 0.26
DROWSY_FRAMES = 30  # Reduced time: ~1 second at 30 FPS
frame_counter = 0
alarm_on = False

# Check model file
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.isfile(predictor_path):
    raise FileNotFoundError("Download and extract 'shape_predictor_68_face_landmarks.dat' from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

# Initialize
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if len(faces) == 0:
        cv2.putText(frame, "FACE NOT DETECTED!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        if not alarm_on:
            alarm_on = True
            play_alarm()
    else:
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            left_eye = draw_eye(frame, landmarks, 36, 42)
            right_eye = draw_eye(frame, landmarks, 42, 48)

            if len(left_eye) == 0 or len(right_eye) == 0:
                cv2.putText(frame, "EYE NOT DETECTED!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                if not alarm_on:
                    alarm_on = True
                    play_alarm()
                continue

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            ear = (left_ear + right_ear) / 2.0
            ear = round(ear, 2)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= DROWSY_FRAMES:
                    cv2.putText(frame, "DROWSY!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                    cv2.putText(frame, "Are you Sleepy?", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    if not alarm_on:
                        alarm_on = True
                        play_alarm()
            else:
                frame_counter = 0
                if alarm_on:
                    stop_alarm()
                    alarm_on = False

            cv2.putText(frame, f"EAR: {ear}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Are you Sleepy?", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        if alarm_on:
            stop_alarm()
        break

cap.release()
cv2.destroyAllWindows()
