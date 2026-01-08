# create_clean_chat.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import os
import re

# 1. Load Translation AI
print("Loading Translation AI...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "mps"
model.to(device)

# 2. Load Alpaca (WE KNOW THIS WORKS)
print("Loading Alpaca...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 3. Helpers
def translate_to_odia(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        force_lang_id = tokenizer.convert_tokens_to_ids("ory_Orya")
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=force_lang_id, 
                max_length=64
            )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except:
        return ""

def to_odinglish(text):
    if not text: return ""
    try:
        text = sanscript.transliterate(text, sanscript.ORIYA, sanscript.ITRANS)
        return text.lower().strip().replace("..", ".")
    except:
        return ""

def is_safe_chat(text):
    text = text.lower()
    
    # FILTER 1: NO NUMBERS (Kills math)
    if re.search(r'\d', text): return False
    
    # FILTER 2: NO LIST/CODE WORDS (Kills lists)
    banned_words = ['list', 'step', 'code', 'function', 'solve', 'calculate', 
                    'math', 'output', 'program', 'html', 'python', 'enumerate']
    if any(w in text for w in banned_words): return False
    
    # FILTER 3: LENGTH (Keep it conversational)
    if len(text) < 5 or len(text) > 80: return False
    
    return True

# 4. Generate
output_file = "clean_chat_data.txt"
if os.path.exists(output_file): os.remove(output_file)

print("Mining Alpaca for Pure Chat (Filtering out Math/Lists)...")
buffer = []
count = 0

# Scan 10,000 items to find the good stuff
for i in tqdm(range(10000)):
    item = dataset[i]
    en_input = item['instruction']
    en_reply = item['output']
    
    # Only process if BOTH input and reply are safe
    if is_safe_chat(en_input) and is_safe_chat(en_reply):
        try:
            # Translate
            od_input = translate_to_odia(en_input)
            od_reply = translate_to_odia(en_reply)
            
            # Transliterate
            odinglish_input = to_odinglish(od_input)
            odinglish_reply = to_odinglish(od_reply)
            
            if odinglish_input and odinglish_reply:
                entry = f"User: {odinglish_input}\nSovogpt: {odinglish_reply}\n<|endoftext|>\n"
                buffer.append(entry)
                count += 1
        except:
            continue
            
    if len(buffer) >= 50:
        with open(output_file, "a", encoding="utf-8") as f:
            f.writelines(buffer)
        buffer = []

if buffer:
    with open(output_file, "a", encoding="utf-8") as f:
        f.writelines(buffer)

print(f"\nSUCCESS! Extracted {count} lines of Safe Chat.")