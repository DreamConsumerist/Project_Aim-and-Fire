import cv2
import mediapipe as mp
import torch
from collections import deque
import numpy as np
import json

# --- Setup Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

cap = cv2.VideoCapture(1)

# --- Load trained TCN model ---
from tcn_model_creation import TCNGestureClassifier  # adjust import if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = 63
num_classes = 3
model = TCNGestureClassifier(num_features=num_features, num_classes=num_classes)
model.load_state_dict(torch.load("tcn_gesture_model_best.pth", map_location=device))
model.to(device)
model.eval()

# --- Label map ---
label_map = {0: "Idle", 1: "Aim", 2: "Fire"}

# --- Sequence & smoothing buffers ---
SEQ_LEN = 10          # input sequence length for model
SMOOTH_LEN = 5        # rolling buffer for smoothing predictions
seq_buffer = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=SMOOTH_LEN)

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

# --- Main loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
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

    cv2.putText(frame, "Press ESC to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
