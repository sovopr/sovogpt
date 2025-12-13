from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
import torch
import json
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

print("--- Step 1: Generating Semantic Data ---")

# 1. Load Translator
print("Loading Translator (NLLB)...")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

def translate_to_odinglish(text):
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        out = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=64)
    odia = trans_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip()
    except: return odia

# 2. Define Concepts in English (The AI will translate these)
# 0=Identity, 1=Chat, 2=Weather, 3=Search
concepts = {
    0: ["who are you", "what is your name", "introduce yourself", "tell me your name", "identify yourself"],
    1: ["hello", "how are you", "what are you doing", "good morning", "how is it going", "help me", "can you help"],
    2: ["weather in london", "temperature today", "is it raining", "forecast for tomorrow", "climate report"],
    3: ["capital of france", "who is the president", "price of bitcoin", "news today", "meaning of life", "population of india"]
}

text_list = []
label_list = []

print("Synthesizing Training Examples...")
for label, phrases in concepts.items():
    for phrase in phrases:
        # Add English version (for robustness)
        text_list.append(phrase)
        label_list.append(label)
        
        # Add Odinglish version (for target language)
        # We generate this dynamically so we don't hardcode "kemiti acha"
        odinglish = translate_to_odinglish(phrase)
        if odinglish:
            print(f"   Generated: {odinglish} (Label: {label})")
            text_list.append(odinglish)
            label_list.append(label)

# 3. Save
with open("router_data.json", "w") as f:
    json.dump({"text": text_list, "label": label_list}, f)

print("\n[SUCCESS] Data generated. Proceed to Step 2.")