# generate_intelligence.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import random
import re
import os

# 1. Setup Translator (NLLB)
print("Loading Translator...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "mps"
model.to(device)

# 2. Helpers
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

buffer = []

# --- PART A: TEACHING TOOL USE (The Agentic Layer) ---
print("Synthesizing Tool-Use Scenarios...")

# We define intents, not just keywords
search_templates = [
    "what is the weather in {loc}?",
    "weather forecast for {loc}",
    "temperature in {loc}",
    "who is {person}?",
    "tell me about {person}",
    "where is {loc}?",
    "capital of {loc}",
    "latest news about {topic}",
    "meaning of {word}",
    "who won the {event}?",
    "price of {item}",
]

locations = ["bhubaneswar", "odisha", "india", "delhi", "mumbai", "london", "puri", "cuttack", "japan", "usa"]
people = ["narendra modi", "naveen patnaik", "mahatma gandhi", "virat kohli", "elon musk"]
topics = ["cricket", "politics", "technology", "movies", "science"]
items = ["gold", "bitcoin", "iphone", "petrol"]

# Generate 3,000 Tool Examples
for _ in tqdm(range(3000)):
    template = random.choice(search_templates)
    
    # Fill template dynamically
    if "{loc}" in template: 
        entity = random.choice(locations)
        q = template.format(loc=entity)
        search_cmd = f"{entity} weather" if "weather" in template else f"{entity} location"
    elif "{person}" in template: 
        entity = random.choice(people)
        q = template.format(person=entity)
        search_cmd = f"{entity} who is"
    elif "{topic}" in template:
        entity = random.choice(topics)
        q = template.format(topic=entity)
        search_cmd = f"{entity} news"
    elif "{item}" in template:
        entity = random.choice(items)
        q = template.format(item=entity)
        search_cmd = f"{entity} price"
    else:
        q = "what is ai?"
        search_cmd = "ai definition"

    # Translate Question
    od_q = to_odinglish(translate(q))
    
    if od_q:
        # THE AGENTIC PATTERN: 
        # User asks -> Sovogpt outputs specialized token <<SEARCH>> + query
        entry = f"User: {od_q}\nSovogpt: <<SEARCH>> {search_cmd}\n<|endoftext|>\n"
        buffer.append(entry)

# --- PART B: TEACHING CONVERSATION (The Human Layer) ---
print("Processing DailyDialog (Human Chat)...")
try:
    # We use a slice to be fast
    ds_chat = load_dataset("daily_dialog", split="train[:2000]", trust_remote_code=True)
except:
    # Fallback if daily_dialog script fails again (Alpaca Backup)
    print("DailyDialog failed, using filtered Alpaca...")
    ds_chat = load_dataset("tatsu-lab/alpaca", split="train[:2000]")

for item in tqdm(ds_chat):
    # Handle different dataset structures
    if 'dialog' in item:
        q, a = item['dialog'][0], item['dialog'][1]
    else:
        q, a = item['instruction'], item['output']
        
    # STRICT FILTERS (The "Anti-Hallucination" Shield)
    # 1. No Numbers (Prevents Math/List loops)
    if re.search(r'\d', q) or re.search(r'\d', a): continue
    # 2. Length (Keep it snappy)
    if len(q) > 60 or len(a) > 60: continue
    
    try:
        od_q = to_odinglish(translate(q))
        od_a = to_odinglish(translate(a))
        
        if od_q and od_a:
            entry = f"User: {od_q}\nSovogpt: {od_a}\n<|endoftext|>\n"
            buffer.append(entry)
    except: continue

# --- PART C: IDENTITY ---
print("Injecting Self-Awareness...")
identity = [
    ("tu kie?", "mu sovogpt, apananka ai sahayaka."),
    ("tumara nama kana?", "mora nama sovogpt."),
    ("tame kemiti acha?", "mu bhala achi!"),
    ("namaskar", "namaskar!"),
]
for q, a in identity:
    for _ in range(50): # Repetition reinforces memory
        buffer.append(f"User: {q}\nSovogpt: {a}\n<|endoftext|>\n")

# Save
random.shuffle(buffer)
with open("agent_training_data.txt", "w", encoding="utf-8") as f:
    f.writelines(buffer)

print(f"DONE. Created {len(buffer)} training examples.")