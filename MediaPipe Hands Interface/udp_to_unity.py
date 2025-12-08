import sys
print(sys.executable)
import socket
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from tcn_model_def import TCNGestureClassifier
import os

UDP_IP = "127.0.0.1"
UDP_PORT = 5555
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # turn off auto exposure
#cap.set(cv2.CAP_PROP_FPS, 60)             # lock frame rate
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)     # fixed brightness

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "tcn_gesture_model_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = 63
num_classes = 3
model = TCNGestureClassifier(num_features=num_features, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Label map ---
label_map = {0: "Idle", 1: "Aim", 2: "Fire"}

# --- Sequence & smoothing buffers ---
SEQ_LEN = 10          # input sequence length for model
SMOOTH_LEN = 5        # rolling buffer for smoothing predictions
seq_buffer = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=SMOOTH_LEN)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def to_vec(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)

# --- Landmark normalization ---
def normalize_hand_landmarks(hand_landmarks, scale=True):
    lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = lm_array[0].copy()
    lm_array -= wrist
    if scale:
        max_dist = np.max(np.linalg.norm(lm_array, axis=1))
        if max_dist > 0:
            lm_array /= max_dist
    return lm_array.flatten().tolist()

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
    pred_label = "None"
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
            normalized = normalize_hand_landmarks(hand_landmarks)
            seq_buffer.append(normalized)

        if len(seq_buffer) == SEQ_LEN:
            seq_tensor = torch.tensor([seq_buffer], dtype=torch.float32).to(device)
            with torch.no_grad():
                output = model(seq_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                pred_buffer.append(pred_class)
            if pred_buffer:
                # majority vote smoothing
                majority_class = max(set(pred_buffer), key=pred_buffer.count)
                pred_label = label_map[majority_class]
                cv2.putText(frame, f"Predicted: {pred_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            seq_buffer.clear()
            pred_buffer.clear()  # clear smoothing when hand lost

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

        message = f"{pred_label},{wrist.x},{wrist.y},{indexTIP.x},{indexTIP.y}"

    cv2.imshow("Hand Classification", frame)
    if message != "":
        encoded_message = message.encode('utf-8')
        sock.sendto(encoded_message, (UDP_IP, UDP_PORT))
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break
sock.close()