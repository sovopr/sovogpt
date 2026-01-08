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

print("\n[Initializing Sovogpt Semantic Architecture...]")

# 1. Load Router
print("1. Loading Semantic Router...")
try:
    router = SetFitModel.from_pretrained("./sovogpt_router")
    intents = {0: "IDENTITY", 1: "CHAT", 2: "WEATHER", 3: "SEARCH"}
except:
    print("Error: Router not found. Run Step 2 first.")
    exit()

# 2. Load Tools
print("2. Loading Translation Tools...")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

# 3. Load Chat Brain
print("3. Loading Chat Brain...")
model_path = "./sovogpt_instruct" 
try:
    chat_model = LlamaForCausalLM.from_pretrained(model_path)
    chat_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    generator = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer, device="mps")
except:
    print("Error: sovogpt_instruct not found.")
    exit()

# --- INTELLIGENT TOOLS ---

def search_web(query):
    # Clean query
    clean_q = " ".join([w for w in query.split() if w.lower() not in ['kana', 'kie', 'kauthare', 're', 'kuade', 'kebe', 'tame', 'apana']])
    print(f"   [Tool] Googling: {clean_q}")
    try:
        results = DDGS().text(clean_q, max_results=3)
        if results:
            for res in results:
                # Filter out garbage (Questions, Options)
                if "..." not in res['body'][:10] and "Question" not in res['body']: 
                    return res['body'].split('.')[0] + "."
        return None
    except: return None

def get_weather(text):
    # Extract City
    clean = text.lower().translate(str.maketrans('', '', string.punctuation))
    ignore = ['kimiti', 'achhi', 'kana', 're', 'in', 'at', 'temperature', 'paga', 'weather', 'ra']
    words = [w for w in clean.split() if w not in ignore]
    city = words[0] if words else "Bhubaneswar"
    
    print(f"   [Tool] Weather API for: {city}")
    try:
        url = f"https://wttr.in/{city}?format=%l:+%t+%C"
        response = requests.get(url, timeout=5) # 5s timeout
        if response.status_code == 200:
            return response.text.strip()
    except:
        print("   [Tool] API timed out. Switching to Google Search...")
        # FALLBACK: If API fails, Google it!
        return search_web(f"weather in {city}")

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
    prompt = f"User: {text}\nSovogpt:"
    res = generator(
        prompt, 
        max_new_tokens=60, 
        do_sample=True, 
        temperature=0.6, 
        repetition_penalty=1.2,
        pad_token_id=50256
    )[0]['generated_text']
    
    ans = res.split("Sovogpt:")[-1].split("User:")[0].strip()
    ans = ans.replace("<|endoftext", "").replace("|>", "").strip()
    return ans if len(ans) > 2 else "mu bujhi parili nahin."

# --- MAIN LOOP ---
print("\n--- Sovogpt Semantic is Online ---")

while True:
    u = input("\nYou: ")
    if u == "quit": break
    
    # 1. PREDICT INTENT
    pred = router.predict([u])[0]
    intent = intents[pred.item()]
    
    # 2. LOGIC OVERRIDES (The "Smart" Layer)
    u_lower = u.lower()
    
    # Override A: If it mentions "You/Tame/Tu", it is PERSONAL (Chat/Identity), never SEARCH.
    # This prevents searching for "Where do you live" -> "Top 10 songs"
    is_personal = any(x in u_lower for x in ['tame', 'tu ', 'apana', 'tumara', 'tor '])
    
    if is_personal:
        if "weather" in u_lower or "news" in u_lower:
            # "Tame weather janicha ki?" -> Still a question, let it pass
            pass
        elif "kie" in u_lower or "na" in u_lower:
            intent = "IDENTITY"
        else:
            intent = "CHAT"

    # 3. ROUTE
    if intent == "IDENTITY":
        # Handle location questions about self
        if "kuade" in u_lower or "kauthare" in u_lower or "live" in u_lower:
            print("Sovogpt: mu kompyutar bhitare rahe.") # "I live inside computer"
        else:
            print("Sovogpt: mu sovogpt, apananka odia ai assistant.")
        
    elif intent == "WEATHER":
        fact = get_weather(u)
        # Check if the result is English (API) or Odinglish (Fallback translation)
        # Just translate it to be safe
        if fact and any(c.isdigit() for c in fact):
             # If it looks like API output (Bhubaneswar: +32C), just show it or translate parts
             print(f"Sovogpt: {translate(fact)} ({fact})")
        else:
             print(f"Sovogpt: {translate(fact)}")
        
    elif intent == "SEARCH":
        fact = search_web(u)
        if fact:
            print(f"Sovogpt: {translate(fact)}")
        else:
            print("Sovogpt: kichi tathya milila nahin.")
            
    elif intent == "CHAT":
        print(f"Sovogpt: {chat(u)}")