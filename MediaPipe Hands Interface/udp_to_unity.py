import sys
print(sys.executable)
import socket
import random
import cv2
import mediapipe as mp

UDP_IP = "127.0.0.1"
UDP_PORT = 5555

print("hello world")
cap = cv2.VideoCapture(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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
        message = f"{hand_landmarks.landmark[8].x},{hand_landmarks.landmark[8].y}"
    cv2.imshow("Hand Classification", frame)
    if message != "":
        encoded_message = message.encode('utf-8')
        sock.sendto(encoded_message, (UDP_IP, UDP_PORT))
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
sock.close()