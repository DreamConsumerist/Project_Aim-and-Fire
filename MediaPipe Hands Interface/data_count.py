import pandas as pd
df = pd.read_csv("hand_gestures.csv", header=None)
print(df[0].value_counts())