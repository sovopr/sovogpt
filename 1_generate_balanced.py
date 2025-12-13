import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import random
import re

print("--- Generating Perfectly Balanced Data ---")

# 1. Load Translator
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

def translate(text):
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        gen = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=64)
    odia = trans_tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except: return odia

buffer = []

# --- PART A: 2,500 SEARCH EXAMPLES ---
print("Generating 2,500 Search Examples...")
templates = [
    ("weather in {loc}", "{loc} weather"),
    ("who is {person}", "{person} who is"),
    ("capital of {loc}", "{loc} capital"),
    ("meaning of {word}", "{word} meaning"),
]
entities = {
    "{loc}": ["bhubaneswar", "odisha", "india", "delhi", "mumbai", "london", "puri", "cuttack", "tokyo", "paris"],
    "{person}": ["modi", "naveen patnaik", "virat kohli", "elon musk", "gandhi"],
    "{word}": ["love", "life", "ai", "democracy", "freedom"]
}

for _ in tqdm(range(2500)):
    tmpl, cmd_tmpl = random.choice(templates)
    key = [k for k in entities.keys() if k in tmpl][0]
    val = random.choice(entities[key])
    
    q_en = tmpl.replace(key, val)
    cmd = cmd_tmpl.replace(key, val)
    
    q_od = translate(q_en) # Translate question
    if q_od:
        # TARGET: <<SEARCH>> command
        buffer.append(f"User: {q_od}\nSovogpt: <<SEARCH>> {cmd}\n<|endoftext|>\n")

# --- PART B: 2,500 CHAT EXAMPLES ---
print("Generating 2,500 Chat Examples...")
# Use Alpaca because it downloads reliably
ds = load_dataset("tatsu-lab/alpaca", split="train[:10000]") # Scan more to find good ones

chat_count = 0
for item in tqdm(ds):
    if chat_count >= 2500: break
    
    q, a = item['instruction'], item['output']
    
    # Strict filter: No numbers, short length
    if re.search(r'\d', q) or re.search(r'\d', a): continue
    if len(q) > 60 or len(a) > 60: continue
    
    q_od = translate(q)
    a_od = translate(a)
    
    if q_od and a_od:
        # TARGET: Normal text
        buffer.append(f"User: {q_od}\nSovogpt: {a_od}\n<|endoftext|>\n")
        chat_count += 1

# --- PART C: IDENTITY (Crucial Fix) ---
print("Injecting Identity...")
ids = [
    ("tu kie?", "mu sovogpt."),
    ("tumara nama kana?", "mora nama sovogpt."),
    ("tame kemiti acha?", "mu bhala achi."),
    ("namaskar", "namaskar!"),
]
for q, a in ids:
    for _ in range(50): # 200 total lines
        buffer.append(f"User: {q}\nSovogpt: {a}\n<|endoftext|>\n")

random.shuffle(buffer)
with open("balanced_training.txt", "w", encoding="utf-8") as f:
    f.writelines(buffer)

print(f"DONE. Created {len(buffer)} Balanced Examples.")
