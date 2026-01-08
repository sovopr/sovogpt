# prepare_chat_data.py
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import re

# 1. Setup the Transliteration Function
def to_odinglish(text):
    if not text: return ""
    try:
        # Convert Odia script to English script (ITRANS scheme)
        text = sanscript.transliterate(text, sanscript.ORIYA, sanscript.ITRANS)
        text = text.lower()
        # Clean up common artifacts
        text = text.replace("..", ".").replace("  ", " ")
        return text.strip()
    except:
        return ""

pairs = []

# --- PHASE 1: Manual Chit-Chat (The "Manners" Layer) ---
print("Phase 1: Injecting Manual Chit-Chat...")
# Since we don't have a book, we hardcode the basics so it isn't rude.
# These are mixed Odia/English patterns.
manual_chats = [
    ("namaskar", "namaskar! apana kemiti achanti?"),
    ("kemiti acha?", "mu bhala achi. apana kemiti achanti?"),
    ("hi", "hallo! mu sovogpt."),
    ("hello", "namaskar!"),
    ("tu kie?", "mu jane ai assistant, mora nama sovogpt."),
    ("tumara nama kana?", "mora nama sovogpt."),
    ("what is your name?", "mora nama sovogpt."),
    ("kana karucha?", "mu apananka saha katha heuchi."),
    ("khana khailu?", "na, mu robot, mu khana khae nahin."),
    ("dhanyabad", "apananku swagata."),
    ("bye", "bidaya!"),
    ("subha sakala", "subha sakala! apananka dina mangalamaya heu."),
    ("subha ratri", "subha ratri."),
    ("odisha kauthare achi?", "odisha bharatara purba bhagare achi."),
    ("bhubaneswar kana?", "bhubaneswar odishara rajadhani."),
    ("mu tumaku bhala paye", "dhanyabad! mu bi tumaku sahajya karibaku bhala paye."),
]

for q, a in manual_chats:
    # We add these 50 times each so the model REALLY learns them
    for _ in range(50):
        pairs.append(f"User: {q}\nSovogpt: {a}")

# --- PHASE 2: Wikipedia Question Mining (The "Knowledge" Layer) ---
print("Phase 2: Mining Questions from Wikipedia...")
# We use the official repo that worked for you before
ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.or", split="train")

count_wiki = 0
# We will scan more articles this time since we lost the book data
limit = 10000 

for item in tqdm(ds_wiki):
    text = item['text']
    # Split article into sentences based on '।' (Odia full stop) or '?'
    sentences = re.split(r'[।?]', text)
    
    for i in range(len(sentences) - 1):
        s1 = sentences[i].strip()
        s2 = sentences[i+1].strip()
        
        # Strategy: Look for Odia Question Words
        # if sentence 1 has a question word, assume sentence 2 is the answer.
        question_words = ["କିଏ", "କଣ", "କାହିଁକି", "କେବେ", "କେଉଁ", "କେତେ", "କିପରି"]
        is_question = any(x in s1 for x in question_words)
        
        # Length check to avoid garbage
        if is_question and len(s1) > 10 and len(s2) > 10:
            q = to_odinglish(s1 + "?") 
            a = to_odinglish(s2 + ".")
            
            # Formatting check: Ensure it looks like Odinglish (ASCII)
            # This filters out lines that failed transliteration
            if q and a:
                pairs.append(f"User: {q}\nSovogpt: {a}")
                count_wiki += 1
            
    if count_wiki > limit: break 

print(f"Total conversational pairs: {len(pairs)}")

# Save to file
with open("real_chat_dataset.txt", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(pair + "\n<|endoftext|>\n")

print("Saved to real_chat_dataset.txt")