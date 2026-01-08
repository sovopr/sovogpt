# create_synthetic_data.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indic_transliteration import sanscript
from tqdm import tqdm
import os

# 1. Load the "Teacher" (Translation Model)
print("Loading Translation AI (NLLB-200)...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "mps" 
model.to(device)

# 2. Load the Source Data
print("Loading English Instructions...")
ds_alpaca = load_dataset("tatsu-lab/alpaca", split="train")

# 3. Helpers
def translate_to_odia(text):
    # FIX: Use convert_tokens_to_ids instead of lang_code_to_id
    # 'ory_Orya' is the code for Odia
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        # Get the token ID for Odia correctly
        force_lang_id = tokenizer.convert_tokens_to_ids("ory_Orya")
        
        with torch.no_grad():
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=force_lang_id, 
                max_length=128
            )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def to_odinglish(text):
    if not text: return ""
    try:
        text = sanscript.transliterate(text, sanscript.ORIYA, sanscript.ITRANS)
        return text.lower().strip().replace("..", ".")
    except:
        return ""

# 4. The Factory Loop
print("Generating Odinglish Chat Data (This should take ~1-2 seconds per item)...")

output_file = "synthetic_chat.txt"
# Clear previous empty file
if os.path.exists(output_file):
    os.remove(output_file)

buffer = []
success_count = 0

# Let's try 100 items first to make sure it works. Increase range(100) to range(500) if successful.
for i in tqdm(range(5000)): 
    item = ds_alpaca[i]
    en_inst = item['instruction']
    en_out = item['output']
    
    if len(en_inst) > 100 or len(en_out) > 200: continue

    try:
        # Translate
        od_inst = translate_to_odia(en_inst)
        od_out = translate_to_odia(en_out)
        
        if not od_inst or not od_out: continue

        # Transliterate
        odinglish_inst = to_odinglish(od_inst)
        odinglish_out = to_odinglish(od_out)
        
        formatted = f"User: {odinglish_inst}\nSovogpt: {odinglish_out}\n<|endoftext|>\n"
        buffer.append(formatted)
        success_count += 1
        
        if len(buffer) >= 5:
            with open(output_file, "a", encoding="utf-8") as f:
                f.writelines(buffer)
            buffer = []
            
    except Exception as e:
        print(f"Skipped: {e}")

# Save remaining
if buffer:
    with open(output_file, "a", encoding="utf-8") as f:
        f.writelines(buffer)

print(f"REAL Success! {success_count} lines saved to {output_file}")