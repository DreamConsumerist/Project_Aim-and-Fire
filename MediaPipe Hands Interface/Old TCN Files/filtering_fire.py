import json

with open("fire_sequences_augmented.json", "r") as f:
    fire_sequences = json.load(f)

# Keep only sequences that are exactly 10 frames
filtered_fire = [seq for seq in fire_sequences if len(seq["landmarks"]) == 10]

print(f"Filtered Fire sequences: {len(filtered_fire)}")

with open("fire_sequences_10_filtered.json", "w") as f:
    json.dump(filtered_fire, f)