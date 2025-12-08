import json

# Load the static sequences
with open("static_sequences_10.json", "r") as f:
    static_sequences = json.load(f)

# Separate sequences by label
idle_sequences = [seq for seq in static_sequences if seq["label"] == "Idle"]
aim_sequences = [seq for seq in static_sequences if seq["label"] == "Aim"]

# Save them into separate files
with open("idle_sequences_10.json", "w") as f:
    json.dump(idle_sequences, f)

with open("aim_sequences_10.json", "w") as f:
    json.dump(aim_sequences, f)

print(f"Idle: {len(idle_sequences)} sequences")
print(f"Aim: {len(aim_sequences)} sequences")