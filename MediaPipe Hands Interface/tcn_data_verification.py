import json
import numpy as np
import matplotlib.pyplot as plt

with open("hand_data.json") as f:
    data = json.load(f)

# Step through each frame
for i, frame in enumerate(data):
    landmarks = np.array(frame["landmarks"])
    label = frame["label"]

    plt.clf()
    plt.scatter(landmarks[:,0], landmarks[:,1])
    plt.title(f"Frame {i}, Label: {label}")
    plt.draw()
    plt.show()
    input("Press Enter for next frame...")