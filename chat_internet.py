# chat_internet.py (Final V3 - Smart Search)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, LlamaForCausalLM, PreTrainedTokenizerFast
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import re
import warnings

# 1. Silence all warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore") 

# --- LOAD TRANSLATOR ---
print("Loading Translator...")
trans_model_name = "facebook/nllb-200-distilled-600M"
try:
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
    device = "mps"
    trans_model.to(device)
except Exception as e:
    print(f"Error loading translator: {e}")
    exit()

# --- LOAD SOVOGPT ---
print("Loading Sovogpt...")
sovogpt_path = "./sovogpt_v5_safe" 

try:
    chat_model = LlamaForCausalLM.from_pretrained(sovogpt_path)
    chat_tokenizer = PreTrainedTokenizerFast.from_pretrained(sovogpt_path)
    generator = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer, device=device)
except:
    print(f"Warning: Could not load {sovogpt_path}. Local brain offline.")
    generator = None

# --- SMART SEARCH LOGIC ---

def clean_query_for_search(text):
    """Optimizes the query for a search engine"""
    text = text.lower()
    
    # 1. Strip Odia fillers
    stopwords = ['kana', 'kie', 'kauthare', 'kebe', 'kipari', 'achi', 'achhi', 're', 'ku', 'ra', 'ate']
    words = [w for w in text.split() if w not in stopwords]
    clean_text = " ".join(words)
    
    # 2. Add context boosters (The Fix for 'Quiz' results)
    if 'weather' in clean_text:
        clean_text += " current temperature forecast"
    elif 'capital' in clean_text:
        clean_text += " capital city name"
        
    return clean_text

def get_internet_answer(query):
    ddgs = DDGS()
    clean_q = clean_query_for_search(query)
    
    print(f"   [System] Googling: '{clean_q}'...")
    
    try:
        # Get top 3 results
        results = ddgs.text(clean_q, max_results=3)
        if not results: return None
        
        # Filter: Find the best result
        for res in results:
            body = res['body']
            
            # Skip "Quiz" or "Option" garbage
            if "(a)" in body or "Question" in body or "..." in body[:10]:
                continue
                
            # If asking for weather, ensure we found numbers
            if 'weather' in query and not re.search(r'\d', body):
                continue
                
            # Return the first good sentence
            first_sentence = body.split('. ')[0] + "."
            return first_sentence
            
        # Fallback: Just take the first one if filtering failed
        return results[0]['body'].split('. ')[0] + "."
        
    except Exception as e:
        print(f"   [Error] {e}")
        return None

def english_to_odinglish(text):
    if not text: return "kichi tathya milila nahin."
    
    # Translate
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    force_lang_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        translated = trans_model.generate(**inputs, forced_bos_token_id=force_lang_id, max_length=128)
    odia = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    # Transliterate
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except:
        return odia

def get_sovogpt_reply(user_input):
    if not generator: return "Error."
    # Hardcoded Greeting Check
    if user_input.lower().strip() in ['hi', 'hello', 'namaskar', 'kemiti acha', 'kemiti achanti?']:
        return "namaskar! apana kemiti achanti?"

    base_prompt = f"User: {user_input}\nSovogpt:"
    response = generator(
        base_prompt, max_new_tokens=40, do_sample=True, temperature=0.5, 
        pad_token_id=50256, eos_token_id=50256
    )
    return response[0]['generated_text'].split("Sovogpt:")[-1].split("User:")[0].strip()

# --- MAIN ---
print("\n--- Sovogpt (Hybrid V3) is ready! ---")
print("(Type 'quit' to exit)")

search_triggers = ['weather', 'news', 'capital', 'meaning', 'score', 'price', 'who', 'what', 'where']

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit": break
    
    # Router
    should_search = False
    if any(w in user_input.lower() for w in search_triggers): should_search = True
    elif len(user_input.split()) > 4: should_search = True 
        
    if should_search:
        fact = get_internet_answer(user_input)
        if fact:
            print(f"   [System] Fact: {fact}")
            print(f"Sovogpt: {english_to_odinglish(fact)}")
        else:
            print("   [System] Search failed. Asking local brain.")
            print(f"Sovogpt: {get_sovogpt_reply(user_input)}")
    else:
        print(f"Sovogpt: {get_sovogpt_reply(user_input)}")