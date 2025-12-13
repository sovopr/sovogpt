import torch
from transformers import pipeline, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from setfit import SetFitModel
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import re
import requests
import string
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("\n[Initializing Sovogpt Dual-Brain System...]")

# 1. Load Cortex
print("1. Loading Cortex...")
try:
    router = SetFitModel.from_pretrained("./sovogpt_cortex")
    intents = {0: "IDENTITY", 1: "CHAT", 2: "WEATHER", 3: "SEARCH"}
except:
    print("Error: Cortex not found.")
    exit()

# 2. Load Translator
print("2. Loading Translator...")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

# 3. Load Chat Brain
print("3. Loading Sovogpt Instruct...")
model_path = "./sovogpt_instruct" 
try:
    chat_model = LlamaForCausalLM.from_pretrained(model_path)
    chat_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    generator = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer, device="mps")
except:
    print("Error: sovogpt_instruct not found.")
    exit()

# --- TOOLS ---

def get_weather(text):
    clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    ignore = ['kimiti', 'achhi', 'kana', 're', 'in', 'at', 'temperature', 'paga', 'weather', 'ra', 'varsha', 'aaji', 'heba', 'ki', 'mote', 'lage', 'bhala', 'kete']
    words = [w for w in clean.split() if w not in ignore]
    city = words[0] if words else "Bhubaneswar"
    
    print(f"   [Tool] Weather API for: {city}")
    try:
        url = f"https://wttr.in/{city}?format=%l:+%t+%C"
        return requests.get(url, timeout=3).text.strip()
    except: return None

def search_web(query):
    # 1. Clean basic stopwords
    clean_text = query.lower().translate(str.maketrans('', '', string.punctuation))
    stopwords = ['kana', 'kie', 'kiye', 'kauthare', 're', 'kuade', 'kebe', 'kete', 'kan', 'ra', 'ku', 'achhi', 'kimiti', 'ka', 'ta', 'au']
    words = [w for w in clean_text.split() if w not in stopwords]
    clean_q = " ".join(words)
    
    # 2. SMART EXPANSION (The Fix)
    # Expand abbreviations so Google understands specific intent
    if "cm " in clean_q or clean_q.endswith("cm"): 
        clean_q = clean_q.replace("cm", "current chief minister name")
    elif "pm " in clean_q or clean_q.endswith("pm"): 
        clean_q = clean_q.replace("pm", "current prime minister name")
    elif "capital" in clean_q:
        clean_q += " capital city name"
    elif "population" in clean_q:
        clean_q += " current population count"
        
    print(f"   [Tool] Googling: {clean_q}")
    
    try:
        # Get 3 results to verify
        results = DDGS().text(clean_q, max_results=3)
        if not results: return None
        
        for res in results:
            body = res['body']
            title = res['title']
            full_text = (title + " " + body).lower()
            
            # 3. RELEVANCE CHECK
            # If asking for Minister, ensure result mentions "Minister" or a Name
            if "minister" in clean_q and "minister" not in full_text:
                continue # Skip generic state wiki pages
                
            # Filter Garbage
            if "..." not in body[:10] and "Question" not in body: 
                print(f"   [Debug] Found: {body[:60]}...")
                return body.split('.')[0] + "."
                
        # Fallback
        return results[0]['body'].split('.')[0] + "."
    except: return None

def translate(text):
    if not text: return "Kichi milila nahin."
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        out = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=128, no_repeat_ngram_size=2)
    odia = trans_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except: return odia

def chat(text):
    system_context = f"User: namaskar\nSovogpt: namaskar! mu sovogpt.\nUser: {text}\nSovogpt:"
    res = generator(
        system_context, 
        max_new_tokens=50, 
        do_sample=True, 
        temperature=0.5, 
        repetition_penalty=1.2, 
        pad_token_id=50256
    )[0]['generated_text']
    
    ans = res.split("Sovogpt:")[-1].split("User:")[0].strip()
    ans = ans.replace("<|endoftext", "").strip()
    if len(ans) < 2 or "adhara" in ans: return "mu bujhi parili nahin."
    return ans

# --- MAIN LOOP ---
print("\n--- Sovogpt System is Online ---")

while True:
    u = input("\nYou: ")
    if u == "quit": break
    
    # 1. CORTEX DECISION
    pred = router.predict([u])[0]
    intent = intents[pred.item()]
    
    # 2. LOGIC OVERRIDES
    if "bhala lage" in u.lower() or "love" in u.lower() or "like" in u.lower(): intent = "CHAT"
        
    # 3. EXECUTION
    if intent == "IDENTITY":
        print("Sovogpt: mu sovogpt, apananka odia ai assistant.")
    elif intent == "WEATHER":
        fact = get_weather(u)
        if fact: print(f"Sovogpt: {translate(fact)} ({fact})")
        else: print("Sovogpt: Paga tathya milila nahin.")
    elif intent == "SEARCH":
        fact = search_web(u)
        if fact: print(f"Sovogpt: {translate(fact)}")
        else: print("Sovogpt: kichi tathya milila nahin.")
    elif intent == "CHAT":
        print(f"Sovogpt: {chat(u)}")