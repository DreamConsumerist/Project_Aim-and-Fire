import cv2
import mediapipe as mp
import json
import time
import numpy as np
from pynput import keyboard
import os
import gc

# --- Load previous data if it exists ---
static_sequences_file = "Old TCN Files/static_gestures.json"
dynamic_sequences_file = "Old TCN Files/dynamic_gestures.json"

if os.path.exists(static_sequences_file):
    with open(static_sequences_file, "r") as f:
        static_sequences = json.load(f)
else:
    static_sequences = []

if os.path.exists(dynamic_sequences_file):
    with open(dynamic_sequences_file, "r") as f:
        dynamic_sequences = json.load(f)
else:
    dynamic_sequences = []

# --- Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)

# --- Key tracking ---
key_label_map = {
    '1': {"label": "Idle", "type": "static"},
    '2': {"label": "Aim", "type": "static"},
    '3': {"label": "Fire", "type": "dynamic"}
}

pressed_keys = set()
current_label = None
current_type = None

def on_press(key):
    global current_label, current_type
    try:
        k = key.char
        if k in key_label_map:
            pressed_keys.add(k)
            current_label = key_label_map[k]["label"]
            current_type = key_label_map[k]["type"]
    except AttributeError:
        pass

def on_release(key):
    global current_label, current_type
    try:
        k = key.char
        if k in pressed_keys:
            pressed_keys.remove(k)
        if not pressed_keys:
            current_label = None
            current_type = None
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --- Data storage ---
dynamic_buffer = []
dynamic_buffer_label = None

# --- Landmark normalization ---
def normalize_hand_landmarks(hand_landmarks, scale=True):
    """
    Normalize MediaPipe hand landmarks for TCN input.
    Centers on wrist and optionally scales by hand size.
    Returns flat list of 63 floats.
    """
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

    # --- Process detected hands ---
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_label is not None:
                normalized = normalize_hand_landmarks(hand_landmarks)
                if current_type == "static":
                    static_sequences.append({
                        "landmarks": normalized,
                        "label": current_label
                    })
                elif current_type == "dynamic":
                    if not dynamic_buffer:
                        dynamic_buffer_label = current_label
                    dynamic_buffer.append(normalized)

    # --- Handle key release or no hands ---
    if not pressed_keys and dynamic_buffer:
        dynamic_sequences.append({
            "landmarks": dynamic_buffer.copy(),
            "label": dynamic_buffer_label
        })
        dynamic_buffer.clear()
        dynamic_buffer_label = None

    # --- Display info ---
    display_label = current_label if current_label else "Not recording"
    cv2.putText(frame, f"Label: {display_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Record", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        if dynamic_buffer:
            dynamic_sequences.append({
                "landmarks": dynamic_buffer.copy(),
                "label": dynamic_buffer_label
            })
            dynamic_buffer.clear()
            dynamic_buffer_label = None
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
listener.stop()

# --- Save data ---
with open("Old TCN Files/static_gestures.json", "w") as f:
    json.dump(static_sequences, f)

with open("Old TCN Files/dynamic_gestures.json", "w") as f:
    json.dump(dynamic_sequences, f)
gc.collect()