import json
import random

# --- Load JSON files ---
with open("idle_sequences_10.json", "r") as f:
    idle_sequences = json.load(f)

with open("aim_sequences_10.json", "r") as f:
    aim_sequences = json.load(f)

with open("fire_sequences_10_filtered.json", "r") as f:
    fire_sequences = json.load(f)

# --- Sample 125 sequences from each ---
idle_sample = random.sample(idle_sequences, 74)  # all 125
aim_sample = random.sample(aim_sequences, 74)
fire_sample = fire_sequences

# --- Combine and shuffle ---
all_sequences = idle_sample + aim_sample + fire_sample
random.shuffle(all_sequences)

# --- Save combined JSON ---
with open("balanced_sequences_take2.json", "w") as f:
    json.dump(all_sequences, f)