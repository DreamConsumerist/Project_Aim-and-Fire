import numpy as np
import json

SEQUENCE_LENGTH = 10  # target length

# --- Load data ---
with open("dynamic_gestures.json", "r") as f:
    fire_sequences = json.load(f)

with open("static_gestures.json", "r") as f:
    static_frames = json.load(f)

# --- Process fire sequences ---
processed_fire = []

for seq in fire_sequences:
    frames = seq["landmarks"]
    if len(frames) < SEQUENCE_LENGTH:
        continue  # discard sequences too short
    # Sliding window for sequences longer than SEQUENCE_LENGTH
    for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH // 2):
        chunk = frames[start:start + SEQUENCE_LENGTH]
        processed_fire.append({"landmarks": chunk, "label": seq["label"]})

print(f"Fire sequences after splitting: {len(processed_fire)}")

# --- Process static frames ---
# Convert static frames into overlapping sequences of SEQUENCE_LENGTH
processed_static = []
labels = set([f["label"] for f in static_frames])

for label in labels:
    frames = [f["landmarks"] for f in static_frames if f["label"] == label]
    for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH // 2):
        chunk = frames[start:start + SEQUENCE_LENGTH]
        processed_static.append({"landmarks": chunk, "label": label})

print(f"Static sequences after splitting: {len(processed_static)}")

# --- Save processed data ---
with open("dynamic_sequences_10.json", "w") as f:
    json.dump(processed_fire, f)

with open("static_sequences_10.json", "w") as f:
    json.dump(processed_static, f)