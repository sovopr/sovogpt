import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, LlamaForCausalLM, PreTrainedTokenizerFast
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import re
import warnings

# 1. System Setup
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("\n[Initializing Sovogpt Agent...]")

# --- COMPONENT 1: TRANSLATOR (NLLB) ---
print("Loading Translation Engine...")
try:
    trans_model_name = "facebook/nllb-200-distilled-600M"
    trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
    trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
    device = "mps"
    trans_model.to(device)
except Exception as e:
    print(f"Error loading NLLB: {e}")
    exit()

# --- COMPONENT 2: LOCAL BRAIN ---
print("Loading Local Personality...")
sovogpt_path = "./sovogpt_v5_safe" # Or sovogpt_complete

try:
    chat_model = LlamaForCausalLM.from_pretrained(sovogpt_path)
    chat_tokenizer = PreTrainedTokenizerFast.from_pretrained(sovogpt_path)
    generator = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer, device=device)
except:
    print(f"Warning: Could not load {sovogpt_path}. Local brain offline.")
    generator = None

# --- COMPONENT 3: UNRESTRICTED SEARCH ---

def clean_query(text):
    """Simple cleaner. Doesn't over-think it."""
    text = text.lower()
    # Remove obvious Odinglish grammar to help the search engine
    stopwords = ['kana', 'kie', 'kauthare', 'kuade', 'kebe', 'kipari', 'achi', 'achhi', 're', 'ku', 'ra', 'ate']
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)

def search_web(raw_query):
    ddgs = DDGS()
    core_query = clean_query(raw_query)
    
    # Simple Context Boosters
    search_term = core_query
    if 'weather' in core_query: search_term += " current temperature celsius"
    if 'capital' in core_query: search_term += " capital city"
    
    print(f"   [Agent] Googling: '{search_term}'...")
    
    try:
        # Just get the top result. Don't be picky.
        results = ddgs.text(search_term, max_results=1)
        if results:
            text = results[0]['body']
            # Just take the first sentence to keep translation clean
            return text.split('. ')[0] + "."
        return None
    except Exception as e:
        print(f"   [Agent] Search Error: {e}")
        return None

def english_to_odinglish(text):
    if not text: return "kichi tathya milila nahin."
    
    # Translate En -> Or
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    force_lang_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        translated = trans_model.generate(**inputs, forced_bos_token_id=force_lang_id, max_length=128)
    odia_text = trans_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    # Transliterate Or -> En
    try:
        return sanscript.transliterate(odia_text, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except:
        return odia_text

def get_local_reply(user_input):
    if not generator: return "Error."
    
    # 1. Soft Hardcode (Only for exact greetings)
    # We do this because 120M models struggle with consistency
    u = user_input.lower().strip()
    if u in ['hi', 'hello', 'namaskar']: 
        return "namaskar! apana kemiti achanti?"
    
    # 2. Neural Generation (With Anti-Looping Fixes)
    base_prompt = f"User: {user_input}\nSovogpt:"
    response = generator(
        base_prompt, 
        max_new_tokens=50, 
        do_sample=True, 
        temperature=0.4,       # Lower temp = less gibberish
        top_k=40,
        repetition_penalty=1.5, # HIGH penalty stops ",,,,,," loops
        no_repeat_ngram_size=2, # Forces it to never repeat words
        pad_token_id=50256, 
        eos_token_id=50256
    )
    
    raw = response[0]['generated_text'].split("Sovogpt:")[-1].split("User:")[0].strip()
    
    # 3. Garbage Filter
    if len(raw) < 2 or "ekagpt" in raw or ",," in raw:
        return "mu bujhi parili nahin. (I didn't understand)"
        
    return raw

# --- MAIN LOOP ---
print("\n--- Sovogpt (Final) is ready! ---")
print("(Type 'quit' to exit)")

# EXPANDED Trigger List
# Any sentence with these words goes to the Internet
search_triggers = [
    'weather', 'news', 'capital', 'meaning', 'score', 'price', 
    'who', 'what', 'where', 'when', 'population',
    'kana', 'kie', 'kauthare', 'kuade', 'kebe' # Odinglish Question Words
]

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit": break
    
    # Router Logic
    should_search = False
    
    # If it has a question word, SEARCH. Don't trust the local brain.
    if any(w in user_input.lower() for w in search_triggers): 
        should_search = True
    elif "?" in user_input:
        should_search = True
    elif len(user_input.split()) > 4: 
        should_search = True
        
    if should_search:
        fact = search_web(user_input)
        if fact:
            print(f"   [Agent] Found: {fact}")
            translated_fact = english_to_odinglish(fact)
            print(f"Sovogpt: {translated_fact}")
        else:
            # If internet fails, say sorry. Don't hallucinate.
            print("   [Agent] No results found.")
            print("Sovogpt: kichi tathya milila nahin.")
            
    else:
        # Chit Chat
        print(f"Sovogpt: {get_local_reply(user_input)}")