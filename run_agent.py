# run_agent.py
import torch
from transformers import pipeline, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import re

logging.getLogger("transformers").setLevel(logging.ERROR)

print("\n[Initializing Sovogpt Agent...]")

# 1. Load Translation Tools
print("Loading Translator...")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

# 2. Load Your New Agent Model
print("Loading Neural Brain...")
model_path = "./sovogpt_agent_model" # <--- The one you just trained
try:
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")
except:
    print("Error: Model not found.")
    exit()

def execute_search(query):
    print(f"   [AI TOOL USE] Searching for: '{query}'")
    try:
        # We search exactly what the model requested
        results = DDGS().text(query, max_results=1)
        if results:
            return results[0]['body'].split('.')[0] + "."
    except:
        return None

def translate_reply(text):
    # Translate English Fact -> Odia -> Odinglish
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        out = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=128)
    odia = trans_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except:
        return odia

print("\n--- Sovogpt is Online ---")
print("(Type 'quit' to exit)")

while True:
    u = input("\nYou: ")
    if u.lower() == "quit": break
    
    # 1. Ask the Brain
    prompt = f"User: {u}\nSovogpt:"
    
    # We use low temperature (0.1) so it strictly follows the <<SEARCH>> format if needed
    output = generator(prompt, max_new_tokens=40, temperature=0.1, do_sample=True, pad_token_id=50256)[0]['generated_text']
    answer = output.split("Sovogpt:")[-1].split("User:")[0].strip()
    
    # 2. Check if Brain used a Tool
    if "<<SEARCH>>" in answer:
        # Extract the query
        query = answer.replace("<<SEARCH>>", "").replace("]]", "").strip()
        
        # Run Tool
        fact = execute_search(query)
        
        if fact:
            print(f"   [Fact]: {fact}")
            translated = translate_reply(fact)
            print(f"Sovogpt: {translated}")
        else:
            print("Sovogpt: kichi tathya milila nahin.")
            
    else:
        # Just Chat
        print(f"Sovogpt: {answer}")