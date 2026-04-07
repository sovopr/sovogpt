import torch
from transformers import pipeline, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import re
import string
import requests # <--- The fix for accurate weather

logging.getLogger("transformers").setLevel(logging.ERROR)

print("\n[Initializing Sovogpt Ultimate...]")

# --- 1. LOAD TRANSLATOR (With Anti-Loop settings) ---
print("1. Loading Translation Engine...")
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

# --- 2. LOAD BRAIN ---
print("2. Loading Sovogpt Instruct...")
# We use 'instruct' because it is your most polite model
model_path = "./sovogpt_instruct"
try:
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")
except:
    print(f"Error: Could not load {model_path}. Check if folder exists.")
    exit()

# --- 3. THE TOOLKIT ---

def get_weather(city):
    """
    Directly fetches weather instead of Googling it. 
    Fixes the 'Train vs Rain' bug.
    """
    print(f"   [Tool] Fetching live weather for: {city}...")
    try:
        # wttr.in returns pure text: "Bhubaneswar: 32C Sunny"
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        return None
    return None

def clean_query(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Expanded Stopwords List (Includes variations like 'kimiti')
    stopwords = [
        'kana', 'kie', 'kauthare', 'kuade', 'kebe', 'kipari', 'kimiti', 'kemiti',
        'achi', 'achhi', 'achha', 're', 'ku', 'ra', 'ate', 'tame', 'apana', 'mu', 'pls', 'please'
    ]
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)

def validate_search_result(text):
    """Prevents Exam Papers and 'Train' errors"""
    text = text.lower()
    
    # 1. Reject Exam Papers
    if any(x in text for x in ["(a)", "(b)", "question", "marks", "attempt"]):
        return False
        
    # 2. Reject non-English (Chinese/Russian spam)
    if len(re.sub(r'[ -~]', '', text)) > 5:
        return False
        
    return True

def smart_search(query):
    # If weather, USE THE API (Bypass Google)
    if 'weather' in query or 'temperature' in query:
        city = query.replace('weather', '').replace('temperature', '').strip()
        if not city: city = "Bhubaneswar" # Default
        return get_weather(city)

    # Otherwise, Search Web
    clean_q = clean_query(query)
    print(f"   [Cortex] Googling: '{clean_q}'...")
    
    try:
        results = DDGS().text(clean_q, max_results=5)
        if not results: return None
        
        for res in results:
            body = res['body']
            if validate_search_result(body):
                return body.split('. ')[0] + "."
    except: return None

def translate_to_odinglish(text):
    # En -> Or
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    
    with torch.no_grad():
        # ANTI-LOOP FIX: no_repeat_ngram_size prevents "relabaira relabaira"
        out = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=128, no_repeat_ngram_size=2)
        
    odia = trans_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    
    # Or -> Odinglish
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except: return odia

def generate_chat(user_input):
    # 1. Identity Guard
    if any(x in user_input.lower() for x in ['tu kie', 'tame kie', 'tumara na']):
        return "mu sovogpt, apananka odia ai assistant."
    
    # 2. Greeting Guard (Because 120M models are inconsistent)
    if any(x in user_input.lower() for x in ['hi', 'hello', 'namaskar', 'kemiti']):
        return "namaskar! apana kemiti achanti?"

    # 3. Neural Fallback
    prompt = f"User: {user_input}\nSovogpt:"
    response = generator(
        prompt, 
        max_new_tokens=40, 
        do_sample=True, 
        temperature=0.6,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    raw = response[0]['generated_text'].split("Sovogpt:")[-1].split("User:")[0].strip()
    
    if len(raw) < 2 or "<<" in raw:
        return "mu bujhi parili nahin."
    return raw

# --- MAIN LOOP ---
print("\n--- Sovogpt Ultimate is Online ---")
print("(Weather API + Smart Search + Neural Chat)")

triggers = ['weather', 'news', 'capital', 'meaning', 'score', 'who', 'what', 'where', 'population']

while True:
    u = input("\nYou: ")
    if u.lower() == "quit": break
    
    # Logic
    if any(w in u.lower() for w in triggers) or "?" in u:
        fact = smart_search(u)
        if fact:
            print(f"   [Fact]: {fact}")
            print(f"Sovogpt: {translate_to_odinglish(fact)}")
        else:
            print("Sovogpt: kichi tathya milila nahin.")
    else:
        print(f"Sovogpt: {generate_chat(u)}")