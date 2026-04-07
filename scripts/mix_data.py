# mix_data.py
import random

# 1. The Manners (We need to remind the model of these constantly)
manual_chats = [
    ("namaskar", "namaskar! apana kemiti achanti?"),
    ("kemiti acha?", "mu bhala achi. apana kemiti achanti?"),
    ("tu kie?", "mu jane ai assistant, mora nama sovogpt."),
    ("tumara nama kana?", "mora nama sovogpt."),
    ("kana karucha?", "mu apananka saha katha heuchi."),
    ("khana khailu?", "na, mu robot, mu khana khae nahin."),
    ("dhanyabad", "apananku swagata."),
    ("bye", "bidaya!"),
    ("odisha ra rajadhani kana?", "odisha ra rajadhani bhubaneswar ate."), # Added this specific one for you
    ("india ra rajadhani kana?", "india ra rajadhani new delhi ate."),
]

print("Loading Synthetic Data...")
# Load the smart data you generated
with open("synthetic_chat.txt", "r", encoding="utf-8") as f:
    synthetic_lines = f.readlines()

print(f"Found {len(synthetic_lines)} lines of logic.")

# 2. The Mixer
# We create 200 copies of the manners to balance out the 5000 lines of facts.
manners_lines = []
for q, a in manual_chats:
    entry = f"User: {q}\nSovogpt: {a}\n<|endoftext|>\n"
    for _ in range(200): 
        manners_lines.append(entry)

# 3. Combine and Shuffle
combined = synthetic_lines + manners_lines
random.shuffle(combined)

# 4. Save
with open("final_mix.txt", "w", encoding="utf-8") as f:
    f.writelines(combined)

print(f"Done! Saved {len(combined)} lines to final_mix.txt")