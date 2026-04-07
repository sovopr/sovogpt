import logging
import os
import re
import warnings
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

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
]

SYSTEM_RULES = (
    "You are Sovogpt, a highly conversational Odia+English AI friend. "
    "Always reply in Roman script (English letters only), using natural Odinglish. "
    "Be empathetic, short, and very human-like."
)

BAD_MARKERS = ["<<", "instruction:", "user:", "sovogpt:", "endoftext", "<|", "|>"]
STOPWORDS = {
    "mu",
    "tu",
    "tume",
    "tame",
    "apana",
    "apanaka",
    "apananka",
    "tamara",
    "toro",
    "tumara",
    "mora",
    "your",
    "naa",
    "nama",
    "name",
    "kana",
    "ki",
    "re",
    "ta",
    "na",
    "naa",
    "what",
    "is",
    "are",
    "the",
}

DECODE_CONFIGS: List[Dict[str, float]] = [
    {"temperature": 0.75, "top_p": 0.9},
    {"temperature": 0.9, "top_p": 0.95},
    {"temperature": 0.65, "top_p": 0.85},
    {"temperature": 0.8, "top_p": 0.92},
]


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


def load_model() -> Tuple[LlamaForCausalLM, PreTrainedTokenizerFast, str]:
    for path in MODEL_CANDIDATES:
        try:
            model = LlamaForCausalLM.from_pretrained(path)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
            return model, tokenizer, path
        except Exception:
            continue
    raise RuntimeError("No model found in known paths")


def build_prompt(history: List[Tuple[str, str]], user_text: str) -> str:
    prompt = f"<|im_start|>system\n{SYSTEM_RULES}<|im_end|>\n"
    for old_user, old_reply in history[-4:]:
        prompt += f"<|im_start|>user\n{old_user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{old_reply}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def sanitize_reply(text: str) -> str:
    chunk = text.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
    chunk = chunk.replace("<|endoftext|>", " ")
    chunk = to_roman_odia(chunk)
    return WS_RE.sub(" ", chunk).strip()


def generate_reply(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    device: str,
    history: List[Tuple[str, str]],
    user_text: str,
) -> str:
    prompt = build_prompt(history, user_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Note: skip_special_tokens=False so we can parse <|im_end|>
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    raw = full[len(prompt) :] if full.startswith(prompt) else full
    return sanitize_reply(raw)


def main() -> None:
    print("\n[Initializing Sovogpt Roman Odia Chat...]")
    model, tokenizer, model_path = load_model()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_path}")
    print("Generation mode: reward-scored multi-sampling (no template hardcoding)")
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
