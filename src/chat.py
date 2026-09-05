# chat.py (Robust Final Version for Sovogpt)
import re
import logging
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

# Silence warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

print("Loading Sovogpt (Smart Edition)...")

model_path = "./sovogpt_agent_model"
device = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"Sovogpt loaded successfully on {device.upper()}!")
except Exception as e:
    print(f"Error loading {model_path}: {e}")
    exit(1)

SYSTEM_RULES = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
)

print("\n--- Sovogpt is ready! ---")
print("(Type 'quit' to exit)\n")

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
        
    # ChatML formatted prompt
    prompt = f"<|im_start|>system\n{SYSTEM_RULES}<|im_end|>\n"
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
    answer = raw.split("<|im_end|>")[0].split("<|im_start|>")[0].replace("<|endoftext|>", " ").strip()
    answer = re.sub(r"[^a-z0-9 .,?!'/-]+", " ", answer.lower())
    answer = re.sub(r"\s+", " ", answer).strip()
    
    if len(answer) < 3:
        answer = "mu apananka katha bujhiparuchi. kichi au pacharantu."
        
    print(f"Sovogpt: {answer}\n")
    history.append((user_input, answer))