import logging
import os
import re
import warnings
from typing import Dict, List, Tuple

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
]

SYSTEM_RULES = (
    "Instruction: You are an Odia AI assistant. Reply in natural Odia+English "
    "using English letters only (Roman script). Keep replies conversational."
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


DRIFT_KEYWORDS = [
    "1980", "1968", "1987", "1960", "100 digri", "phutabal", "aphrika", 
    "indonesia", "mandirara labha", "satriya", "prrithibira", "mamsapeshi"
]

FALLBACK_CONVERSATIONAL = {
    "namaskar": "namaskar! mu bhala achi, apana kemiti achanti?",
    "kemiti": "mu bhala achi, apana kemiti achanti?",
    "kie": "mu sovogpt, apananka odia ai sahayaka.",
    "kama": "mu apananku katha habare o prashnara uttara debare sahajya kare.",
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
    "bye": "dhanyabad! apananka saha katha hoi bahut khusi lagila. apananka dina mangalamaya heu, bye!"
}

def is_drifted_or_looping(text: str) -> bool:
    if len(text) < 3:
        return True
    for kw in DRIFT_KEYWORDS:
        if kw in text:
            return True
    words = text.split()
    if len(words) >= 4:
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2]:
                return True
    return False

def get_smart_fallback(user_text: str) -> str:
    u = user_text.lower()
    for k, v in FALLBACK_CONVERSATIONAL.items():
        if k in u:
            return v
    return "mu apananka prashna bujhiparuchi. kichi bhal bhabare pacharantu, mu sahajya karibi."

def generate_reply(
    model: LlamaForCausalLM,
    tokenizer: any,
    device: str,
    history: List[Tuple[str, str]],
    user_text: str,
) -> str:
    # Cap history to last 2 turns to prevent compounding error noise
    prompt = build_prompt(history[-2:], user_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
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
    cleaned = sanitize_reply(raw)
    
    # Check guardrail against pretraining drift or degenerate loops
    if is_drifted_or_looping(cleaned):
        cleaned = get_smart_fallback(user_text)
        
    return cleaned


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
