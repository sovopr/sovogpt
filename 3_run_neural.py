import torch
from transformers import pipeline, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from duckduckgo_search import DDGS
from indic_transliteration import sanscript
import logging
import requests

logging.getLogger("transformers").setLevel(logging.ERROR)

print("\n[Initializing Sovogpt Groundbreaking Edition...]")

# 1. Tools
trans_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
trans_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")

def get_weather(city):
    try:
        return requests.get(f"https://wttr.in/{city}?format=%l:+%t+%C", timeout=3).text.strip()
    except: return None

def search_web(query):
    try:
        # Smart Search: Check Weather API first
        if "weather" in query:
            city = query.replace("weather", "").strip()
            w = get_weather(city if len(city)>2 else "Bhubaneswar")
            if w: return w
            
        # Fallback to Google
        res = DDGS().text(query, max_results=1)
        if res: return res[0]['body'].split('.')[0]
    except: return None

def translate(text):
    inputs = trans_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to("mps")
    force_id = trans_tokenizer.convert_tokens_to_ids("ory_Orya")
    with torch.no_grad():
        out = trans_model.generate(**inputs, forced_bos_token_id=force_id, max_length=128)
    odia = trans_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    try:
        return sanscript.transliterate(odia, sanscript.ORIYA, sanscript.ITRANS).lower().strip().replace("..", ".")
    except: return odia

# 2. The Brain
print("Loading Neural Brain...")
model_path = "./sovogpt_groundbreaking"
try:
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    # High repetition penalty ensures it never loops
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="mps")
except:
    print("Error: Model not trained yet.")
    exit()

print("\n--- Sovogpt Online ---")
while True:
    u = input("\nYou: ")
    if u == "quit": break
    
    # Pure Neural Prompt
    prompt = f"User: {u}\nSovogpt:"
    
    output = generator(
        prompt, 
        max_new_tokens=60, 
        do_sample=True, 
        temperature=0.3, # Low temp for accuracy
        repetition_penalty=1.3,
        pad_token_id=50256
    )[0]['generated_text']
    
    answer = output.split("Sovogpt:")[-1].split("User:")[0].strip()
    
    # NEURAL DECISION: Did the model ask for help?
    if "<<SEARCH>>" in answer:
        query = answer.replace("<<SEARCH>>", "").strip()
        print(f"   [Neural Decision]: Searching for '{query}'")
        
        fact = search_web(query)
        if fact:
            print(f"Sovogpt: {translate(fact)}")
        else:
            print("Sovogpt: Kichi tathya milila nahin.")
    else:
        # The model decided to chat
        print(f"Sovogpt: {answer}")