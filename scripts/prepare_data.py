# prepare_data.py
from datasets import load_dataset
from indic_transliteration import sanscript
import re
from tqdm import tqdm

# 1. Download Odia Wikipedia (Using the new secure repo)
print("Downloading Odia Wikipedia...")
# We use 'wikimedia/wikipedia' instead of just 'wikipedia'
# We use the '20231101.or' config which is the verified Odia dump
dataset = load_dataset("wikimedia/wikipedia", "20231101.or", split="train")

# 2. Define the Converter (Odia Script -> English Script)
def to_odinglish(text):
    try:
        # Use 'ORIYA' (the library's internal name for Odia)
        transliterated = sanscript.transliterate(text, sanscript.ORIYA, sanscript.ITRANS)
        return transliterated.lower() 
    except:
        return ""

# 3. Process and Save
print("Converting to Odinglish (this takes time)...")
with open("train_data.txt", "w", encoding="utf-8") as f:
    # Process the first 50,000 articles (or all if you have time)
    # The dataset might be smaller than 50k, so we handle that safely
    count = 0
    limit = 50000
    
    for item in tqdm(dataset): 
        if count >= limit:
            break
            
        text = item["text"].strip()
        if len(text) > 50:
            eng_script = to_odinglish(text)
            # Cleanup common transliteration artifacts
            eng_script = eng_script.replace("..", ".").replace("  ", " ")
            f.write(eng_script + "\n<|endoftext|>\n")
            count += 1

print(f"Done! {count} articles processed. Data saved to train_data.txt")