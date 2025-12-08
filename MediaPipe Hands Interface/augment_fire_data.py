import numpy as np
import random

def augment_fire_sequences(sequences, target_count, jitter=2, noise_std=0.01, frame_dropout_prob=0.1):
    """
    Augment short dynamic gesture sequences to increase dataset size.

    Args:
        sequences: List of dicts, each {"landmarks": [[...]], "label": "Fire"}.
        target_count: Desired total number of sequences after augmentation.
        jitter: Max frames to shift start/end.
        noise_std: Standard deviation of Gaussian noise added to landmarks.
        frame_dropout_prob: Probability to randomly drop a frame.

    Returns:
        Augmented list of sequences.
    """
    augmented = sequences.copy()
    while len(augmented) < target_count:
        seq = random.choice(sequences)
        landmarks = seq["landmarks"]
        seq_len = len(landmarks)

        # --- Temporal jittering ---
        start_shift = random.randint(0, min(jitter, seq_len-1))
        end_shift = random.randint(0, min(jitter, seq_len-1))
        new_seq = landmarks[start_shift: seq_len - end_shift]

        # --- Frame dropout / repetition ---
        frames = []
        for frame in new_seq:
            if random.random() < frame_dropout_prob and len(new_seq) > 1:
                continue  # drop this frame
            frames.append(frame)
        if len(frames) == 0:
            frames = new_seq.copy()

        # --- Add small Gaussian noise ---
        noisy_frames = []
        for frame in frames:
            frame_array = np.array(frame)
            frame_array += np.random.normal(0, noise_std, frame_array.shape)
            noisy_frames.append(frame_array.tolist())

        augmented.append({
            "landmarks": noisy_frames,
            "label": seq["label"]
        })

    return augmented

import json

# Load dynamic sequences
with open("dynamic_sequences_10.json", "r") as f:
    dynamic_sequences = json.load(f)
target_fire_count = 400
fire_sequences_augmented = augment_fire_sequences(dynamic_sequences, target_fire_count)
print(f"Fire sequences augmented: {len(fire_sequences_augmented)}")

with open("fire_sequences_augmented.json", "w") as f:
    json.dump(fire_sequences_augmented, f)