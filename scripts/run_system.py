import logging
import os
import re
import warnings
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

try:
    from indic_transliteration import sanscript

    HAS_TRANSLIT = True
except Exception:
    HAS_TRANSLIT = False

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

ODIA_BLOCK_RE = re.compile(r"[\u0B00-\u0B7F]")
NON_ROMAN_RE = re.compile(r"[^a-z0-9 .,?!'/-]+")
WS_RE = re.compile(r"\s+")

MODEL_CANDIDATES = [
    "./sovogpt_agent_model",
    os.path.join(os.path.dirname(__file__), "..", "sovogpt_agent_model"),
]

SYSTEM_RULES = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
)


def to_roman_odia(text: str) -> str:
    out = text.strip().replace("|", " ").replace("।", ".")
    if HAS_TRANSLIT and ODIA_BLOCK_RE.search(out):
        try:
            out = sanscript.transliterate(out, sanscript.ORIYA, sanscript.ITRANS)
        except Exception:
            pass
    out = out.lower()
    out = NON_ROMAN_RE.sub(" ", out)
    out = WS_RE.sub(" ", out).strip()
    return out


def load_model() -> Tuple[LlamaForCausalLM, any, str]:
    for path in MODEL_CANDIDATES:
        try:
            model = LlamaForCausalLM.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
            return model, tokenizer, path
        except Exception:
            continue
    raise RuntimeError("No model found in known paths")


def build_prompt(history: List[Tuple[str, str]], user_text: str) -> str:
    prompt = f"<|im_start|>system\n{SYSTEM_RULES}<|im_end|>\n"
    if history:
        old_user, old_reply = history[-1]
        clean_reply = old_reply.split(".")[0] + "." if "." in old_reply else old_reply
        prompt += f"<|im_start|>user\n{old_user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{clean_reply}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def sanitize_reply(text: str) -> str:
    chunk = text.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
    chunk = chunk.replace("<|endoftext|>", " ")
    chunk = to_roman_odia(chunk)
    return WS_RE.sub(" ", chunk).strip()


def generate_reply(
    model: LlamaForCausalLM,
    tokenizer: any,
    device: str,
    history: List[Tuple[str, str]],
    user_text: str,
) -> str:
    # Ensure model is on the target device
    target_device = torch.device(device)
    if next(model.parameters()).device != target_device:
        model.to(target_device)
        model.eval()

    # Cap history to last 2 turns to prevent compounding error noise
    prompt = build_prompt(history[-2:], user_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(target_device)
    inputs.pop("token_type_ids", None)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=72,
            do_sample=True,
            temperature=0.35,
            top_p=0.88,
            top_k=40,
            repetition_penalty=1.35,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=im_end_id if im_end_id is not None else tokenizer.eos_token_id,
        )

    # Extract only the newly generated tokens
    new_tokens = out[0][len(inputs["input_ids"][0]):]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return sanitize_reply(raw)


def main() -> None:
    print("\n[Initializing Sovogpt Roman Odia Chat...]")
    model, tokenizer, model_path = load_model()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_path}")
    print("Generation mode: neural ChatML generation with nucleus sampling")
    print("Type 'quit' to exit.")

    history: List[Tuple[str, str]] = []
    while True:
        user_raw = input("\nYou: ").strip()
        if not user_raw:
            continue
        if user_raw.lower() == "quit":
            break

        user_text = to_roman_odia(user_raw)
        reply = generate_reply(model, tokenizer, device, history, user_text)
        print(f"Sovogpt: {reply}")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
