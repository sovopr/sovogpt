# chat.py (Robust Final Version)
from transformers import pipeline, LlamaForCausalLM, PreTrainedTokenizerFast
import torch
import logging

# Silence warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Loading Sovogpt (Smart Edition)...")

model_path = "./sovogpt_complete"

# FIX: We explicitly tell Python "This is a Llama model"
# instead of asking it to guess.
try:
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
except OSError:
    print(f"Error: Could not find {model_path}. Did you finish training?")
    print("Trying to load ./sovogpt_smart (previous version) instead...")
    model_path = "./sovogpt_smart"
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device="mps" 
)

base_prompt = "User: {input}\nSovogpt:"

print("\n--- Sovogpt is ready! ---")
print("(Type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    full_prompt = base_prompt.format(input=user_input)
    
    response = generator(
        full_prompt, 
        max_new_tokens=60,
        num_return_sequences=1, 
        do_sample=True,
        temperature=0.5, 
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=50256,
        eos_token_id=50256
    )
    
    full_text = response[0]['generated_text']
    answer = full_text.split("Sovogpt:")[-1].strip()
    if "User:" in answer:
        answer = answer.split("User:")[0].strip()
        
    print(f"Sovogpt: {answer}\n")