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
    "./sovogpt_agent_model_chatfix",
    "./sovogpt_agent_model_tuned_v2",
    "./sovogpt_agent_model_tuned",
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


def repetition_ratio(words: List[str]) -> float:
    if not words:
        return 1.0
    return 1.0 - (len(set(words)) / len(words))


def token_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_candidate(user_text: str, reply: str) -> float:
    if not reply:
        return -100.0

    rlow = reply.lower()
    score = 0.0

    if any(m in rlow for m in BAD_MARKERS):
        score -= 8.0

    words = reply.split()
    if len(words) < 3:
        score -= 2.0
    elif len(words) <= 40:
        score += 1.0
    else:
        score -= 1.0

    rep = repetition_ratio(words)
    score -= rep * 3.0

    user_tokens = {w for w in token_words(user_text) if len(w) > 2 and w not in STOPWORDS}
    reply_tokens = {w for w in token_words(reply) if len(w) > 2 and w not in STOPWORDS}
    overlap = len(user_tokens & reply_tokens)
    score += min(overlap * 0.25, 1.5)
    if user_tokens and overlap == 0:
        score -= 1.8

    if any(k in user_text for k in ["naa", "nama", "name"]) and not any(
        k in rlow for k in ["mora", "naa", "nama", "name", "sovogpt"]
    ):
        score -= 1.5

    weather_query = any(k in user_text for k in ["temp", "temperature", "weather", "paga", "tapamatra"])
    if weather_query and not any(k in rlow for k in ["temp", "weather", "garam", "thanda", "climate", "humidity"]):
        score -= 1.6

    if reply.endswith("?"):
        score += 0.2

    return score


def generate_candidates(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    device: str,
    prompt: str,
) -> List[str]:
    candidates: List[str] = []
    for cfg in DECODE_CONFIGS:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full = tokenizer.decode(out[0], skip_special_tokens=True)
        raw = full[len(prompt) :] if full.startswith(prompt) else full
        candidates.append(sanitize_reply(raw))
    return candidates


def fallback_reply(user_text: str) -> str:
    if len(user_text.split()) <= 2:
        return "mu bujhi parili nahin, alpa detail re pacha."
    return "bujhili, tume au alpa clear bhabe kuha, mu detail re answer debi."


def generate_reply(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    device: str,
    history: List[Tuple[str, str]],
    user_text: str,
) -> str:
    prompt = build_prompt(history, user_text)
    candidates = generate_candidates(model, tokenizer, device, prompt)
    scored = [(score_candidate(user_text, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_reply = scored[0]
    if best_score < -0.5:
        repair_prompt = (
            f"{SYSTEM_RULES}\n"
            f"User: {user_text}\n"
            "Sovogpt: Reply directly and clearly in 1-2 lines in Odinglish."
        )
        repair_candidates = generate_candidates(model, tokenizer, device, repair_prompt)
        repair_scored = [(score_candidate(user_text, c), c) for c in repair_candidates]
        repair_scored.sort(key=lambda x: x[0], reverse=True)
        rep_score, rep_reply = repair_scored[0]
        if rep_score >= -0.8:
            return rep_reply
        return fallback_reply(user_text)
    return best_reply


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
