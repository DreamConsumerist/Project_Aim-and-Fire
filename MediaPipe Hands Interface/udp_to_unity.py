import sys
print(sys.executable)
import socket
import cv2
import mediapipe as mp
import math
import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def to_vec(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)

UDP_IP = "127.0.0.1"
UDP_PORT = 5555

print("hello world")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # turn off auto exposure
cap.set(cv2.CAP_PROP_FPS, 60)             # lock frame rate
cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)     # fixed brightness

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)
    message = ""
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Pointing direction
        indexMCP = hand_landmarks.landmark[5]
        indexPIP = hand_landmarks.landmark[6]
        indexDIP = hand_landmarks.landmark[7]
        indexTIP = hand_landmarks.landmark[8]
        wrist = hand_landmarks.landmark[0]
        v1 = to_vec(indexMCP, indexPIP)
        v2 = to_vec(indexPIP, indexDIP)
        v3 = to_vec(indexDIP, indexTIP)
        combined = (v1[0] + v2[0] + v3[0], v1[1] + v2[1] + v3[1], v1[2] + v2[2] + v3[2])
        direction = normalize(combined)

        message = f"{wrist.x},{wrist.y},{indexTIP.x},{indexTIP.y}"

    cv2.imshow("Hand Classification", frame)
    if message != "":
        encoded_message = message.encode('utf-8')
        sock.sendto(encoded_message, (UDP_IP, UDP_PORT))
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
sock.close()