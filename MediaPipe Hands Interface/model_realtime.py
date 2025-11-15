import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("finger_gun_cat.keras")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Labels mapping
labels = ['pistol', 'trigger', 'none']  # adjust if your model used different order

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess landmarks relative to wrist
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            wrist = landmarks[0].copy()
            landmarks -= wrist  # center all landmarks relative to wrist
            landmarks = landmarks.flatten().reshape(1, -1)

            # Make prediction
            prediction = model.predict(landmarks, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            # Display label and confidence
            label_text = f"{labels[class_idx]} ({confidence:.2f})"
            cv2.putText(frame, label_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
