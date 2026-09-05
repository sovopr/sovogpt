# chat_internet.py (Final V3 - Smart Search)
import re
import warnings
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaForCausalLM
from indic_transliteration import sanscript

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# 1. Silence all warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore") 

device = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM_RULES = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
)

# --- LOAD TRANSLATOR (LAZY) ---
trans_model = None
trans_tokenizer = None

def get_trans():
    global trans_model, trans_tokenizer
    if trans_model is None:
        try:
            print("Loading Translator...")
            trans_model_name = "facebook/nllb-200-distilled-600M"
            trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
            trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
            trans_model.to(device)
            trans_model.eval()
        except Exception as e:
            print(f"Notice: Translator unavailable ({e})")
    return trans_model, trans_tokenizer

# --- LOAD SOVOGPT ---
print("Loading Sovogpt...")
sovogpt_path = "./sovogpt_agent_model"
model = None
tokenizer = None
im_end_id = None

try:
    model = LlamaForCausalLM.from_pretrained(sovogpt_path)
    tokenizer = AutoTokenizer.from_pretrained(sovogpt_path)
    model.to(device)
    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"Sovogpt loaded on {device.upper()}.")
except Exception as e:
    print(f"Warning: Could not load {sovogpt_path}: {e}")

# --- SMART SEARCH LOGIC ---

def clean_query_for_search(text):
    """Optimizes the query for a search engine"""
    text = text.lower()
    stopwords = ['kana', 'kie', 'kauthare', 'kebe', 'kipari', 'achi', 'achhi', 're', 'ku', 'ra', 'ate']
    words = [w for w in text.split() if w not in stopwords]
    clean_text = " ".join(words)
    
    if 'weather' in clean_text or 'tapamatra' in clean_text:
        clean_text += " current temperature celsius"
    elif 'capital' in clean_text:
        clean_text += " capital city name"
    elif 'news' in clean_text:
        clean_text += " latest news"
        
    return clean_text

def get_internet_answer(query):
    clean_q = clean_query_for_search(query)
    print(f"   [System] Searching: '{clean_q}'...")
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(clean_q, max_results=3))
            if not results:
                return None
            for res in results:
                body = res.get('body', '')
                if "(a)" in body or "Question" in body or "..." in body[:10]:
                    continue
                if ('weather' in query or 'tapamatra' in query) and not re.search(r'\d', body):
                    continue
                if len(body) > 20:
                    return body.split('. ')[0] + "."
            return results[0].get('body', '').split('. ')[0] + "."
    except Exception as e:
        print(f"   [Error] {e}")
        return None

def english_to_odinglish(text):
    if not text:
        return "kichi tathya milila nahin."
    t_model, t_tok = get_trans()
    if t_model is None or t_tok is None:
        return text
        
    inputs = t_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    force_lang_id = t_tok.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        translated = t_model.generate(**inputs, forced_bos_token_id=force_lang_id, max_length=128)
    odia = t_tok.batch_decode(translated, skip_special_tokens=True)[0]
    
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except Exception:
        return odia

IDENTITY_RESPONSES = {
    "hi": "namaskar! apana kemiti achanti?",
    "hello": "namaskar! mu sovogpt.",
    "namaskar": "namaskar! apana kemiti achanti?",
    "kemiti acha": "mu bhala achi, tume kemiti achha?",
    "kemiti achha": "mu bhala achi, tume kemiti achha?",
    "tume kie": "mu sovogpt, apananka odia ai sahayaka.",
    "tumara nama kana": "mora nama sovogpt.",
    "what is your name": "mora nama sovogpt.",
}

def get_sovogpt_reply(user_input, history=None):
    u = user_input.lower().strip()
    for k, v in IDENTITY_RESPONSES.items():
        if k in u and len(u.split()) <= 4:
            return v

    if not model or not tokenizer:
        return "mu sovogpt, apananka odia ai sahayaka."

    prompt = f"<|im_start|>system\n{SYSTEM_RULES}<|im_end|>\n"
    if history:
        for u_old, a_old in history[-2:]:
            prompt += f"<|im_start|>user\n{u_old}<|im_end|>\n<|im_start|>assistant\n{a_old}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.35,
            top_p=0.88,
            top_k=40,
            repetition_penalty=1.35,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=im_end_id if im_end_id is not None else tokenizer.eos_token_id,
        )

    new_tokens = out[0][len(inputs["input_ids"][0]):]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    cleaned = raw.split("<|im_end|>")[0].split("<|im_start|>")[0].replace("<|endoftext|>", " ").strip()
    cleaned = re.sub(r"[^a-z0-9 .,?!'/-]+", " ", cleaned.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if len(cleaned) >= 3 else "mu bujhiparuchi, kichi au pacharantu."

# --- MAIN ---
print("\n--- Sovogpt (Hybrid V3) is ready! ---")
print("(Type 'quit' to exit)\n")

search_triggers = ['weather', 'tapamatra', 'barsha', 'news', 'capital', 'price', 'population', 'who is prime minister']
history = []

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
        break
        
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    
    u_lower = user_input.lower()
    should_search = any(w in u_lower for w in search_triggers)
        
    if should_search:
        fact = get_internet_answer(user_input)
        if fact:
            print(f"   [System] Fact: {fact}")
            translated = english_to_odinglish(fact)
            print(f"Sovogpt: {translated}\n")
            history.append((user_input, translated))
        else:
            print("   [System] Search unavailable. Consulting local brain.")
            reply = get_sovogpt_reply(user_input, history)
            print(f"Sovogpt: {reply}\n")
            history.append((user_input, reply))
    else:
        reply = get_sovogpt_reply(user_input, history)
        print(f"Sovogpt: {reply}\n")
        history.append((user_input, reply))