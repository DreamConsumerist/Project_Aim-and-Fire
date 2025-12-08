import pandas as pd
import numpy as np

# Load your original dataset
df = pd.read_csv("hand_gestures.csv", header=None)

# Split labels and landmarks
labels = df.iloc[:, 0]
landmarks = df.iloc[:, 1:].to_numpy(dtype=np.float32)

# Flip x-coordinates (assuming x is every 3rd element starting at 0)
flipped_landmarks = landmarks.copy()
flipped_landmarks[:, 0::3] = 1.0 - flipped_landmarks[:, 0::3]  # for normalized 0–1 coordinates

# Create a new DataFrame with the same labels
df_flipped = pd.DataFrame(np.column_stack([labels, flipped_landmarks]))

# Append flipped data to original dataset
df_augmented = pd.concat([df, df_flipped], ignore_index=True)

# Shuffle
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save augmented dataset
df_augmented.to_csv("hand_gestures_augmented.csv", index=False, header=False)