import os
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

# 1. System Setup
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

print("\n[Initializing Sovogpt Agent...]")
device = "mps" if torch.backends.mps.is_available() else "cpu"

SYSTEM_RULES = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
)

# --- COMPONENT 1: TRANSLATOR (NLLB - LAZY LOADED) ---
trans_model = None
trans_tokenizer = None

def get_trans_engine():
    global trans_model, trans_tokenizer
    if trans_model is None:
        try:
            print("   [Agent] Loading NLLB Translation Engine on-demand...")
            trans_model_name = "facebook/nllb-200-distilled-600M"
            trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
            trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)
            trans_model.to(device)
            trans_model.eval()
            print("   [Agent] Translation Engine online.")
        except Exception as e:
            print(f"   [Agent] Notice: NLLB translation engine unavailable ({e}).")
    return trans_model, trans_tokenizer

# --- COMPONENT 2: LOCAL BRAIN (SOVOGPT) ---
print("Loading Local SovoGPT Brain...")
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
    print(f"Local Brain online ({sovogpt_path} on {device.upper()}).")
except Exception as e:
    print(f"Warning: Could not load {sovogpt_path}: {e}. Local brain offline.")

# --- COMPONENT 3: SEARCH & EXTRACTION ---

def clean_query(text: str) -> str:
    """Strips common Odia functional particles to optimize search engine retrieval."""
    text = text.lower()
    stopwords = ['kana', 'kie', 'kauthare', 'kuade', 'kebe', 'kipari', 'achi', 'achhi', 're', 'ku', 'ra', 'ate']
    words = [w for w in text.split() if w not in stopwords]
    return " ".join(words)

def search_web(raw_query: str) -> str:
    core_query = clean_query(raw_query)
    search_term = core_query
    if 'weather' in core_query or 'tapamatra' in core_query:
        search_term += " current temperature celsius"
    elif 'capital' in core_query:
        search_term += " capital city name"
    elif 'news' in core_query:
        search_term += " latest news"
        
    print(f"   [Agent] Searching: '{search_term}'...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_term, max_results=3))
            for res in results:
                body = res.get('body', '')
                if "(a)" in body or "Question" in body or "..." in body[:10]:
                    continue
                if ('weather' in raw_query or 'tapamatra' in raw_query) and not re.search(r'\d', body):
                    continue
                if len(body) > 20:
                    first_sentence = body.split('. ')[0] + "."
                    return first_sentence
            if results:
                return results[0].get('body', '').split('. ')[0] + "."
        return None
    except Exception as e:
        print(f"   [Agent] Search Notice: {e}")
        return None

def english_to_odinglish(text: str) -> str:
    if not text:
        return "kichi tathya milila nahin."
    t_model, t_tok = get_trans_engine()
    if t_model is None or t_tok is None:
        return text
    
    inputs = t_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    force_lang_id = t_tok.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        translated = t_model.generate(**inputs, forced_bos_token_id=force_lang_id, max_length=128)
    odia_text = t_tok.batch_decode(translated, skip_special_tokens=True)[0]
    
    try:
        return sanscript.transliterate(odia_text, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except Exception:
        return odia_text

# --- COMPONENT 4: CONVERSATIONAL MEMORY & GENERATION ---

IDENTITY_MAP = {
    "tume kie": "mu sovogpt, apananka odia ai sahayaka.",
    "tumara nama kana": "mora nama sovogpt.",
    "what is your name": "mora nama sovogpt.",
    "who are you": "mu sovogpt, tumara odia ai friend.",
    "namaskar": "namaskar! apana kemiti achanti?",
    "kemiti achha": "mu bhala achi, tume kemiti achha?",
    "kemiti acha": "mu bhala achi, tume kemiti achha?",
    "hi": "hallo! mu sovogpt.",
    "hello": "namaskar! mu sovogpt.",
    "subha sakala": "subha sakala! apananka dina mangalamaya heu.",
    "subha ratri": "subha ratri! shantire souantu.",
    "dhanyabad": "apananku swagata! jebe darkar pacha.",
    "bye": "bidaya! puni dekha haba.",
}

CONVERSATIONAL_SEEDS = {
    "khaiba": "aaji rati re dalma, bhata kimba roti-tarkari khaiparanti. simple au healthy!",
    "gapa": "dharani re gote gaon thila, sethi eka chota pila thila ye ki sabubele gacha lagauthila. dina se gacha falabanta hela o samastanku sahajya kala.",
    "missi": "han nischaya! ame odia o english mix kari katha heipariba (Odinglish).",
    "movie": "tume 'Daman', 'Babushan nka film', kimba kichi bhala Odia classic cinema dekhipariba.",
    "kounthi": "mu cloud re thiba eka odia ai assistant, mora ghara odisha re boli bhabiparanti!",
    "ai": "artificial intelligence mane krutrima budhimatta, yaha computer ku manisha pari bhabi katha heba sikhaye.",
    "sahitya": "odia sahitya bahut prachina o samrudha. sarala das, fakir mohan senapati, o radhanath ray nkara lekha atyanta lokapriya.",
    "khadya": "mu robot, kintu odisha ra pakhala, dalma, o chhena poda mora favourite boli sunichi!",
    "advice": "sabubele positive bhabantu, samayare kama karantu, o aaji tike bishrama niantu.",
    "udas": "udas huantu nahin! tike bhal music sunantu, sanga nka saha katha huantu, sabu thik heijiba.",
}

DRIFT_PATTERNS = ["1980", "1968", "1987", "1960", "phutabal", "aphrika", "indonesia", "mandirara labha", "satriya", "prrithibira"]

def get_local_reply(user_input: str, history: list = None) -> str:
    u = user_input.lower().strip()
    
    # 1. Exact / Partial Identity Match
    for k, v in IDENTITY_MAP.items():
        if k in u and len(u.split()) <= 4:
            return v
            
    # 2. Conversational Intent Grounding
    for k, v in CONVERSATIONAL_SEEDS.items():
        if k in u:
            return v
            
    if not model or not tokenizer:
        return "mu sovogpt, apananka odia ai sahayaka."
        
    # 3. Neural ChatML Generation
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
    
    # Guardrail against pretraining drift or repetitive collapse
    if len(cleaned) < 3 or any(p in cleaned for p in DRIFT_PATTERNS):
        return "mu apananka prashna bujhiparuchi. kichi bhal bhabare pacharantu, mu sahajya karibi."
        
    return cleaned

# --- MAIN AGENT LOOP ---
def main():
    print("\n" + "="*50)
    print("  Sovogpt Hybrid Multi-Agent Chat is ready!")
    print("  Type 'quit' to exit")
    print("="*50 + "\n")
    
    search_keywords = ['weather', 'tapamatra', 'barsha', 'news', 'capital', 'price', 'population', 'who is prime minister']
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
        
        # Router: Explicit real-time factual triggers
        is_factual_query = any(k in u_lower for k in search_keywords)
        # Weather / temperature queries always route to live search
        if ('weather' in u_lower or 'tapamatra' in u_lower or 'temperature' in u_lower):
            is_factual_query = True
            
        if is_factual_query:
            fact = search_web(user_input)
            if fact:
                print(f"   [Agent] Fact Found: {fact}")
                translated = english_to_odinglish(fact)
                print(f"Sovogpt: {translated}\n")
                history.append((user_input, translated))
            else:
                reply = "kichi tathya milila nahin. apana tike bhala bhabare pachariparibe ki?"
                print(f"Sovogpt: {reply}\n")
                history.append((user_input, reply))
        else:
            # Chit-chat / local intelligence
            reply = get_local_reply(user_input, history)
            print(f"Sovogpt: {reply}\n")
            history.append((user_input, reply))

if __name__ == "__main__":
    main()