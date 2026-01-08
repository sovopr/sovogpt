import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import random
import re

print("--- Step 1: Massive Data Generation ---")

# 1. Load Translator
print("Loading Translator...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "mps"
model.to(device)

def translate(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        force_id = tokenizer.convert_tokens_to_ids("ory_Orya")
        with torch.no_grad():
            gen = model.generate(**inputs, forced_bos_token_id=force_id, max_length=64)
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except: return ""

def to_odinglish(text):
    try:
        t = sanscript.transliterate(text, sanscript.ORIYA, sanscript.ITRANS)
        return t.lower().strip().replace("..", ".")
    except: return ""

def noisify(text):
    # Simulates typos and slang to make the model robust
    if random.random() > 0.7: text = text.replace('a', '').replace('e', '') # kmti
    if random.random() > 0.7: text = text.replace('i', 'ee') # kee
    if random.random() > 0.7: text = text.replace('?', '') # no question mark
    return text

buffer = []

# --- PART A: ROBUST TOOL USE (10,000 lines) ---
print("Generating Tool Data...")
locations = ["bhubaneswar", "cuttack", "puri", "delhi", "mumbai", "london", "paris", "tokyo", "odisha", "india", "usa"]
weather_qs = ["weather in", "temperature of", "is it raining in", "forecast for", "climate of", "how is the weather in"]
news_qs = ["news about", "latest on", "updates for", "what is happening in"]
def_qs = ["who is", "what is", "tell me about", "meaning of", "define"]

# We generate massive variations
for _ in tqdm(range(3000)):
    loc = random.choice(locations)
    
    # 1. Weather
    tmpl = random.choice(weather_qs)
    q_en = f"{tmpl} {loc}"
    cmd = f"{loc} weather"
    q_od = to_odinglish(translate(q_en))
    # Add Clean Version
    buffer.append(f"User: {q_od}\nSovogpt: <<SEARCH>> {cmd}\n<|endoftext|>\n")
    # Add Noisy Version
    buffer.append(f"User: {noisify(q_od)}\nSovogpt: <<SEARCH>> {cmd}\n<|endoftext|>\n")

    # 2. News/Facts
    tmpl = random.choice(news_qs)
    q_en = f"{tmpl} {loc}"
    cmd = f"{loc} news"
    q_od = to_odinglish(translate(q_en))
    buffer.append(f"User: {q_od}\nSovogpt: <<SEARCH>> {cmd}\n<|endoftext|>\n")

# --- PART B: ROBUST IDENTITY (5,000 lines) ---
print("Generating Identity Data...")
id_questions = [
    "who are you", "what is your name", "tell me your name", "identify yourself", 
    "are you a robot", "your name", "intro"
]
id_answers = [
    "mu sovogpt, apananka ai sahayaka.", 
    "mora nama sovogpt.", 
    "mu jane odia ai assistant."
]

for _ in tqdm(range(1000)):
    q = random.choice(id_questions)
    a = random.choice(id_answers)
    
    # Translate Q
    q_od = to_odinglish(translate(q))
    
    # Add Clean
    buffer.append(f"User: {q_od}\nSovogpt: {a}\n<|endoftext|>\n")
    # Add Noisy (CRITICAL: Fixes 'tame kiye')
    buffer.append(f"User: {noisify(q_od)}\nSovogpt: {a}\n<|endoftext|>\n")
    
    # Hardcode common failures just to be sure
    buffer.append(f"User: tame kiye?\nSovogpt: {a}\n<|endoftext|>\n")
    buffer.append(f"User: tu kie?\nSovogpt: {a}\n<|endoftext|>\n")

# --- PART C: ROBUST CHIT-CHAT (5,000 lines) ---
print("Generating Chat Data...")
try:
    ds = load_dataset("daily_dialog", split="train[:3000]", trust_remote_code=True)
except:
    ds = load_dataset("tatsu-lab/alpaca", split="train[:3000]")

for item in tqdm(ds):
    if 'dialog' in item: q, a = item['dialog'][0], item['dialog'][1]
    else: q, a = item['instruction'], item['output']
    
    if re.search(r'\d', q) or re.search(r'\d', a): continue
    if len(q) > 50: continue
    
    try:
        q_od = to_odinglish(translate(q))
        a_od = to_odinglish(translate(a))
        if q_od and a_od:
            buffer.append(f"User: {q_od}\nSovogpt: {a_od}\n<|endoftext|>\n")
    except: continue

random.shuffle(buffer)
with open("massive_training_data.txt", "w", encoding="utf-8") as f:
    f.writelines(buffer)

print(f"\n[DONE] Generated {len(buffer)} examples. This file is HUGE.")