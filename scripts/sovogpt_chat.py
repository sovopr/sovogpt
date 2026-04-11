"""
Sovogpt CLI chat using nanochat engine.

Interactive Odinglish conversation powered by a nanochat-trained GPT-2 model.
Uses KV-cached inference for fast generation.

Usage:
    cd sovogpt-main/nanochat_engine
    source .venv/bin/activate
    cd ..
    python scripts/sovogpt_chat.py
"""

import os
import sys
import re
import argparse

# Add nanochat to path
NANOCHAT_DIR = os.path.join(os.path.dirname(__file__), "..", "nanochat_engine")
sys.path.insert(0, NANOCHAT_DIR)

import torch

# ─── Transliteration helpers (from existing sovogpt) ─────────────────────────
try:
    from indic_transliteration import sanscript
    HAS_TRANSLIT = True
except ImportError:
    HAS_TRANSLIT = False

ODIA_BLOCK_RE = re.compile(r"[\u0B00-\u0B7F]")
NON_ROMAN_RE = re.compile(r"[^a-z0-9 .,?!'/-]+")
WS_RE = re.compile(r"\s+")


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sovogpt Chat (nanochat engine)")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional SFT checkpoint tag to load")
    parser.add_argument("--step", type=int, default=None, help="Optional checkpoint step to load")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Inference device (auto prefers MPS, then CPU)",
    )
    args = parser.parse_args()

    # Import nanochat modules
    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.engine import Engine
    from nanochat.checkpoint_manager import load_model

    print("\n╔══════════════════════════════════════════╗")
    print("║   Sovogpt × nanochat — Odinglish Chat   ║")
    print("╚══════════════════════════════════════════╝")

    # Load the SFT model
    device_type = autodetect_device_type() if args.device == "auto" else args.device
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.step)

    # Special tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    # Engine for efficient KV-cached generation
    engine = Engine(model, tokenizer)

    print(f"Model loaded on {device_type.upper()}")
    print("Type 'quit' to exit, 'clear' to reset.\n")

    conversation_tokens = [bos]

    while True:
        try:
            user_raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user_raw:
            continue
        if user_raw.lower() in ("quit", "exit"):
            print("bye!")
            break
        if user_raw.lower() == "clear":
            conversation_tokens = [bos]
            print("--- conversation cleared ---")
            continue

        # Transliterate to clean Roman Odia
        user_text = to_roman_odia(user_raw)

        # Add user tokens
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_text))
        conversation_tokens.append(user_end)

        # Start assistant turn
        conversation_tokens.append(assistant_start)

        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": 128,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }

        response_tokens = []
        print("Sovogpt: ", end="", flush=True)
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        print()

        # Ensure assistant_end is appended
        if not response_tokens or response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)


if __name__ == "__main__":
    main()
