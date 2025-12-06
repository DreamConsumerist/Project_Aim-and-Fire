import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv("hand_gestures.csv", header=None)
# Shuffle the rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Optional: save back to CSV
df.to_csv("hand_gestures_shuffled.csv", index=False, header=False)

X = df.iloc[:, 1:].to_numpy(dtype=np.float32)
labels = df.iloc[:, 0].to_numpy()

label_map = {'pistol': 0, 'trigger': 1, 'none': 2}
y = np.array([label_map[label] for label in labels])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Input: 63 features (21 landmarks x 3 coordinates)
num_features = 63
num_classes = 3  # pistol, trigger, none

model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    Dropout(0.2),  # optional: reduces overfitting
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # softmax for multi-class classification
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # integer labels work directly
    metrics=['accuracy']
)

model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

model.save("finger_gun_cat.keras")